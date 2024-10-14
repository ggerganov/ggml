#!/bin/bash

cp -rpv ../whisper.cpp/ggml/CMakeLists.txt       CMakeLists.txt
cp -rpv ../whisper.cpp/ggml/src/CMakeLists.txt   src/CMakeLists.txt
cp -rpv ../whisper.cpp/ggml/cmake/FindSIMD.cmake cmake/FindSIMD.cmake

cp -rpv ../whisper.cpp/ggml/src/ggml.c              src/ggml.c
cp -rpv ../whisper.cpp/ggml/src/ggml-aarch64.c      src/ggml-aarch64.c
cp -rpv ../whisper.cpp/ggml/src/ggml-aarch64.h      src/ggml-aarch64.h
cp -rpv ../whisper.cpp/ggml/src/ggml-alloc.c        src/ggml-alloc.c
cp -rpv ../whisper.cpp/ggml/src/ggml-backend-impl.h src/ggml-backend-impl.h
cp -rpv ../whisper.cpp/ggml/src/ggml-backend.cpp    src/ggml-backend.cpp
cp -rpv ../whisper.cpp/ggml/src/ggml-blas.cpp       src/ggml-blas.cpp
cp -rpv ../whisper.cpp/ggml/src/ggml-cann/*         src/ggml-cann/
cp -rpv ../whisper.cpp/ggml/src/ggml-cann.cpp       src/ggml-cann.cpp
cp -rpv ../whisper.cpp/ggml/src/ggml-common.h       src/ggml-common.h
cp -rpv ../whisper.cpp/ggml/src/ggml-cuda/*         src/ggml-cuda/
cp -rpv ../whisper.cpp/ggml/src/ggml-cuda.cu        src/ggml-cuda.cu
cp -rpv ../whisper.cpp/ggml/src/ggml-impl.h         src/ggml-impl.h
cp -rpv ../whisper.cpp/ggml/src/ggml-kompute.cpp    src/ggml-kompute.cpp
cp -rpv ../whisper.cpp/ggml/src/ggml-metal.m        src/ggml-metal.m
cp -rpv ../whisper.cpp/ggml/src/ggml-metal.metal    src/ggml-metal.metal
cp -rpv ../whisper.cpp/ggml/src/ggml-quants.c       src/ggml-quants.c
cp -rpv ../whisper.cpp/ggml/src/ggml-quants.h       src/ggml-quants.h
cp -rpv ../whisper.cpp/ggml/src/ggml-rpc.cpp        src/ggml-rpc.cpp
cp -rpv ../whisper.cpp/ggml/src/ggml-sycl/*         src/ggml-sycl/
cp -rpv ../whisper.cpp/ggml/src/ggml-sycl.cpp       src/ggml-sycl.cpp
cp -rpv ../whisper.cpp/ggml/src/ggml-vulkan.cpp     src/ggml-vulkan.cpp
cp -rpv ../whisper.cpp/ggml/src/vulkan-shaders/*    src/vulkan-shaders/

cp -rpv ../whisper.cpp/ggml/include/ggml.h         include/ggml.h
cp -rpv ../whisper.cpp/ggml/include/ggml-alloc.h   include/ggml-alloc.h
cp -rpv ../whisper.cpp/ggml/include/ggml-backend.h include/ggml-backend.h
cp -rpv ../whisper.cpp/ggml/include/ggml-blas.h    include/ggml-blas.h
cp -rpv ../whisper.cpp/ggml/include/ggml-cann.h    include/ggml-cann.h
cp -rpv ../whisper.cpp/ggml/include/ggml-cuda.h    include/ggml-cuda.h
cp -rpv ../whisper.cpp/ggml/include/ggml-kompute.h include/ggml-kompute.h
cp -rpv ../whisper.cpp/ggml/include/ggml-metal.h   include/ggml-metal.h
cp -rpv ../whisper.cpp/ggml/include/ggml-rpc.h     include/ggml-rpc.h
cp -rpv ../whisper.cpp/ggml/include/ggml-sycl.h    include/ggml-sycl.h
cp -rpv ../whisper.cpp/ggml/include/ggml-vulkan.h  include/ggml-vulkan.h

cp -rpv ../whisper.cpp/examples/common.h        examples/common.h
cp -rpv ../whisper.cpp/examples/common.cpp      examples/common.cpp
cp -rpv ../whisper.cpp/examples/common-ggml.h   examples/common-ggml.h
cp -rpv ../whisper.cpp/examples/common-ggml.cpp examples/common-ggml.cpp

cp -rpv ../whisper.cpp/LICENSE                ./LICENSE
cp -rpv ../whisper.cpp/scripts/gen-authors.sh ./scripts/gen-authors.sh
