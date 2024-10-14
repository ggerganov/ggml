#!/bin/bash

cp -rpv ../llama.cpp/ggml/CMakeLists.txt       CMakeLists.txt
cp -rpv ../llama.cpp/ggml/src/CMakeLists.txt   src/CMakeLists.txt
cp -rpv ../llama.cpp/ggml/cmake/FindSIMD.cmake cmake/FindSIMD.cmake

cp -rpv ../llama.cpp/ggml/src/ggml.c              src/ggml.c
cp -rpv ../llama.cpp/ggml/src/ggml-aarch64.c      src/ggml-aarch64.c
cp -rpv ../llama.cpp/ggml/src/ggml-aarch64.h      src/ggml-aarch64.h
cp -rpv ../llama.cpp/ggml/src/ggml-alloc.c        src/ggml-alloc.c
cp -rpv ../llama.cpp/ggml/src/ggml-backend-impl.h src/ggml-backend-impl.h
cp -rpv ../llama.cpp/ggml/src/ggml-backend.cpp    src/ggml-backend.cpp
cp -rpv ../llama.cpp/ggml/src/ggml-blas.cpp       src/ggml-blas.cpp
cp -rpv ../llama.cpp/ggml/src/ggml-cann/*         src/ggml-cann/
cp -rpv ../llama.cpp/ggml/src/ggml-cann.cpp       src/ggml-cann.cpp
cp -rpv ../llama.cpp/ggml/src/ggml-common.h       src/ggml-common.h
cp -rpv ../llama.cpp/ggml/src/ggml-cuda/*         src/ggml-cuda/
cp -rpv ../llama.cpp/ggml/src/ggml-cuda.cu        src/ggml-cuda.cu
cp -rpv ../llama.cpp/ggml/src/ggml-impl.h         src/ggml-impl.h
cp -rpv ../llama.cpp/ggml/src/ggml-kompute.cpp    src/ggml-kompute.cpp
cp -rpv ../llama.cpp/ggml/src/ggml-metal.m        src/ggml-metal.m
cp -rpv ../llama.cpp/ggml/src/ggml-metal.metal    src/ggml-metal.metal
cp -rpv ../llama.cpp/ggml/src/ggml-quants.c       src/ggml-quants.c
cp -rpv ../llama.cpp/ggml/src/ggml-quants.h       src/ggml-quants.h
cp -rpv ../llama.cpp/ggml/src/ggml-rpc.cpp        src/ggml-rpc.cpp
cp -rpv ../llama.cpp/ggml/src/ggml-sycl/*         src/ggml-sycl/
cp -rpv ../llama.cpp/ggml/src/ggml-sycl.cpp       src/ggml-sycl.cpp
cp -rpv ../llama.cpp/ggml/src/ggml-vulkan.cpp     src/ggml-vulkan.cpp
cp -rpv ../llama.cpp/ggml/src/vulkan-shaders/*    src/vulkan-shaders/

cp -rpv ../llama.cpp/ggml/include/ggml.h         include/ggml.h
cp -rpv ../llama.cpp/ggml/include/ggml-alloc.h   include/ggml-alloc.h
cp -rpv ../llama.cpp/ggml/include/ggml-backend.h include/ggml-backend.h
cp -rpv ../llama.cpp/ggml/include/ggml-blas.h    include/ggml-blas.h
cp -rpv ../llama.cpp/ggml/include/ggml-cann.h    include/ggml-cann.h
cp -rpv ../llama.cpp/ggml/include/ggml-cuda.h    include/ggml-cuda.h
cp -rpv ../llama.cpp/ggml/include/ggml-kompute.h include/ggml-kompute.h
cp -rpv ../llama.cpp/ggml/include/ggml-metal.h   include/ggml-metal.h
cp -rpv ../llama.cpp/ggml/include/ggml-rpc.h     include/ggml-rpc.h
cp -rpv ../llama.cpp/ggml/include/ggml-sycl.h    include/ggml-sycl.h
cp -rpv ../llama.cpp/ggml/include/ggml-vulkan.h  include/ggml-vulkan.h

cp -rpv ../llama.cpp/tests/test-opt.cpp           tests/test-opt.cpp
cp -rpv ../llama.cpp/tests/test-grad0.cpp         tests/test-grad0.cpp
cp -rpv ../llama.cpp/tests/test-quantize-fns.cpp  tests/test-quantize-fns.cpp
cp -rpv ../llama.cpp/tests/test-quantize-perf.cpp tests/test-quantize-perf.cpp
cp -rpv ../llama.cpp/tests/test-backend-ops.cpp   tests/test-backend-ops.cpp

cp -rpv ../llama.cpp/LICENSE                ./LICENSE
cp -rpv ../llama.cpp/scripts/gen-authors.sh ./scripts/gen-authors.sh
