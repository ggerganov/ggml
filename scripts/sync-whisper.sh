#!/bin/bash

cp -rpv ../whisper.cpp/ggml/CMakeLists.txt       CMakeLists.txt
cp -rpv ../whisper.cpp/ggml/src/CMakeLists.txt   src/CMakeLists.txt
cp -rpv ../whisper.cpp/ggml/cmake/FindSIMD.cmake cmake/FindSIMD.cmake

cp -rpv ../whisper.cpp/ggml/src/ggml*.c          src/
cp -rpv ../whisper.cpp/ggml/src/ggml*.cpp        src/
cp -rpv ../whisper.cpp/ggml/src/ggml*.h          src/
cp -rpv ../whisper.cpp/ggml/src/gguf*.cpp        src/
cp -rpv ../whisper.cpp/ggml/src/ggml-blas/*      src/ggml-blas/
cp -rpv ../whisper.cpp/ggml/src/ggml-cann/*      src/ggml-cann/
cp -rpv ../whisper.cpp/ggml/src/ggml-cpu/*       src/ggml-cpu/
cp -rpv ../whisper.cpp/ggml/src/ggml-cuda/*      src/ggml-cuda/
cp -rpv ../whisper.cpp/ggml/src/ggml-hip/*       src/ggml-hip/
cp -rpv ../whisper.cpp/ggml/src/ggml-kompute/*   src/ggml-kompute/
cp -rpv ../whisper.cpp/ggml/src/ggml-metal/*     src/ggml-metal/
cp -rpv ../whisper.cpp/ggml/src/ggml-musa/*      src/ggml-musa/
cp -rpv ../whisper.cpp/ggml/src/ggml-opencl/*    src/ggml-opencl/
cp -rpv ../whisper.cpp/ggml/src/ggml-rpc/*       src/ggml-rpc/
cp -rpv ../whisper.cpp/ggml/src/ggml-sycl/*      src/ggml-sycl/
cp -rpv ../whisper.cpp/ggml/src/ggml-vulkan/*    src/ggml-vulkan/

cp -rpv ../whisper.cpp/ggml/include/ggml*.h include/
cp -rpv ../whisper.cpp/ggml/include/gguf*.h include/

cp -rpv ../whisper.cpp/examples/common.h        examples/common.h
cp -rpv ../whisper.cpp/examples/common.cpp      examples/common.cpp
cp -rpv ../whisper.cpp/examples/common-ggml.h   examples/common-ggml.h
cp -rpv ../whisper.cpp/examples/common-ggml.cpp examples/common-ggml.cpp

cp -rpv ../whisper.cpp/LICENSE                ./LICENSE
cp -rpv ../whisper.cpp/scripts/gen-authors.sh ./scripts/gen-authors.sh
