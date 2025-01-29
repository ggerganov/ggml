#!/bin/bash

cp -rpv ../llama.cpp/ggml/CMakeLists.txt       CMakeLists.txt
cp -rpv ../llama.cpp/ggml/src/CMakeLists.txt   src/CMakeLists.txt
cp -rpv ../llama.cpp/ggml/cmake/*              cmake/

cp -rpv ../llama.cpp/ggml/src/ggml*.c          src/
cp -rpv ../llama.cpp/ggml/src/ggml*.cpp        src/
cp -rpv ../llama.cpp/ggml/src/ggml*.h          src/
cp -rpv ../llama.cpp/ggml/src/gguf*.cpp        src/
cp -rpv ../llama.cpp/ggml/src/ggml-blas/*      src/ggml-blas/
cp -rpv ../llama.cpp/ggml/src/ggml-cann/*      src/ggml-cann/
cp -rpv ../llama.cpp/ggml/src/ggml-cpu/*       src/ggml-cpu/
cp -rpv ../llama.cpp/ggml/src/ggml-cuda/*      src/ggml-cuda/
cp -rpv ../llama.cpp/ggml/src/ggml-hip/*       src/ggml-hip/
cp -rpv ../llama.cpp/ggml/src/ggml-kompute/*   src/ggml-kompute/
cp -rpv ../llama.cpp/ggml/src/ggml-metal/*     src/ggml-metal/
cp -rpv ../llama.cpp/ggml/src/ggml-musa/*      src/ggml-musa/
cp -rpv ../llama.cpp/ggml/src/ggml-opencl/*    src/ggml-opencl/
cp -rpv ../llama.cpp/ggml/src/ggml-rpc/*       src/ggml-rpc/
cp -rpv ../llama.cpp/ggml/src/ggml-sycl/*      src/ggml-sycl/
cp -rpv ../llama.cpp/ggml/src/ggml-vulkan/*    src/ggml-vulkan/

cp -rpv ../llama.cpp/ggml/include/ggml*.h include/
cp -rpv ../llama.cpp/ggml/include/gguf*.h include/

cp -rpv ../llama.cpp/tests/test-opt.cpp           tests/test-opt.cpp
cp -rpv ../llama.cpp/tests/test-quantize-fns.cpp  tests/test-quantize-fns.cpp
cp -rpv ../llama.cpp/tests/test-quantize-perf.cpp tests/test-quantize-perf.cpp
cp -rpv ../llama.cpp/tests/test-backend-ops.cpp   tests/test-backend-ops.cpp

cp -rpv ../llama.cpp/LICENSE                ./LICENSE
cp -rpv ../llama.cpp/scripts/gen-authors.sh ./scripts/gen-authors.sh
