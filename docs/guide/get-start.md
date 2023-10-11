# Getting Started

## Installation

### Prerequisites

- A compatible operating system (e.g. Linux, macOS, Windows).
- A compatible C++ compiler that supports at least C++11.
- [VSCode](https://code.visualstudio.com/), or other editor.
- CMake or a compatible build tool for building the project.

### Install with CMake

If you don’t already have CMake installed, see the [CMake installation guide](https://cmake.org/resources).

CMake uses a file named CMakeLists.txt to configure the build system for a project. You’ll use this file to set up your
project and declare a dependency on ggml.

```cmake
cmake_minimum_required(VERSION 3.14)
project(my_project)

# ggml requires at least C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GGML_GIT_COMMIT_HASH ef336850d5bfe8237ebca1ec82cdfb97d78baff1)
set(GGML_GIT_URL https://github.com/ggerganov/ggml.git)

include(FetchContent)

FetchContent_Declare(
        ggml
        GIT_REPOSITORY ${GGML_GIT_URL}
        GIT_TAG ${GGML_GIT_COMMIT_HASH}
)

FetchContent_MakeAvailable(ggml)

# include ggml .h file
include_directories(${ggml_SOURCE_DIR}/include)
```

The above configuration declares a dependency on ggml which is downloaded from GitHub

### Git Submodule

You can add ggml as a submodule of your project. In your project repository:

```bash
git submodule add https://github.com/ggerganov/ggml ggml
```

## Create and run a binary

With ggml declared as a dependency, you can use ggml code within your own project.

As an example, create a file named hello_ggml.cpp in your project directory with the following contents:

```c++

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

const struct ggml_init_params params = {
/*.mem_size   =*/ ggml_tensor_overhead() * 10240,
/*.mem_buffer =*/ nullptr,
/*.no_alloc   =*/ true
};
int main(int argc, char ** argv) {
    // this get time
    ggml_time_init();
    const auto t_main_start_us = ggml_time_us();
    // create ggml ctx
    const auto ctx = ggml_init(params);
    // create ggml cpu backend
    const auto backend = ggml_backend_cpu_init();
    // create a 1d tensor
    const auto tensor_a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    const auto tensor_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    auto tensor_c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    std::vector<float> tensor_data_a = {1.0};
    std::vector<float> tensor_data_b = {2.0};
    std::vector<float> tensor_data_c;
    
    const auto tensor_1d_memory_size = ggml_nbytes(tensor_a) + ggml_nbytes(tensor_b);
    // create a backend buffer (can be in host or device memory)
    const auto tensor_1d_buffer = ggml_backend_alloc_buffer(backend, tensor_1d_memory_size + 256);
    
    // set value
    {
    
    const auto alloc = ggml_allocr_new_from_buffer(tensor_1d_buffer);
    // this updates the pointers in the tensors to point to the correct location in the buffer
    // this is necessary since the ggml_context is .no_alloc == true
    // note that the buffer can actually be a device buffer, depending on the backend
    ggml_allocr_alloc(alloc, tensor_a);
    ggml_allocr_alloc(alloc, tensor_b);
    
    // in cpu we also can do
    // tensor_a->data = &tensor_data_a.data();
    // tensor_b->data = &tensor_data_b.data();
    ggml_backend_tensor_set(tensor_a, tensor_data_a.data(), 0, ggml_nbytes(tensor_a));
    ggml_backend_tensor_set(tensor_b, tensor_data_b.data(), 0, ggml_nbytes(tensor_b));
    ggml_allocr_free(alloc);
    }
    
    // compute
    {
    const auto compute_tensor_buffer = ggml_backend_alloc_buffer(backend, 656480);
    const auto allocr = ggml_allocr_new_from_buffer(tensor_1d_buffer);
    const auto gf = ggml_new_graph(ctx);
    
    // creat forward
    tensor_c = ggml_add(ctx, tensor_a, tensor_b);
    ggml_build_forward_expand(gf, tensor_c);
    
    // allocate tensors
    ggml_allocr_alloc_graph(allocr, gf);
    
    if (ggml_backend_is_cpu(backend)) {
    ggml_backend_cpu_set_n_threads(backend, 1);
    }
    
    ggml_backend_graph_compute(backend, gf);
    
    tensor_data_c.resize(1);
    ggml_backend_tensor_get(gf->nodes[gf->n_nodes - 1], tensor_data_c.data(), 0,
    tensor_data_c.size() * sizeof(float));
    printf("result is = %p \n", tensor_data_c.data());
    }
    
    const auto t_main_end_us = ggml_time_us();
    
    printf("total time = %8.2f ms\n", (t_main_end_us - t_main_start_us) / 1000.0f);
    
    
    ggml_free(ctx);
    ggml_backend_buffer_free(tensor_1d_buffer);
    ggml_backend_free(backend);
}
```