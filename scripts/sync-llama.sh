#!/bin/bash

cp -rpv ../llama.cpp/ggml/CMakeLists.txt       CMakeLists.txt
cp -rpv ../llama.cpp/ggml/src/CMakeLists.txt   src/CMakeLists.txt
cp -rpv ../llama.cpp/ggml/cmake/FindSIMD.cmake cmake/FindSIMD.cmake

cp -rpv ../llama.cpp/ggml/src/ggml*.c          src/
cp -rpv ../llama.cpp/ggml/src/ggml*.cpp        src/
cp -rpv ../llama.cpp/ggml/src/ggml*.h          src/
cp -rpv ../llama.cpp/ggml/src/ggml*.cu         src/
cp -rpv ../llama.cpp/ggml/src/ggml*.m          src/
cp -rpv ../llama.cpp/ggml/src/ggml-amx/*       src/ggml-amx/
cp -rpv ../llama.cpp/ggml/src/ggml-cann/*      src/ggml-cann/
cp -rpv ../llama.cpp/ggml/src/ggml-cuda/*      src/ggml-cuda/
cp -rpv ../llama.cpp/ggml/src/ggml-sycl/*      src/ggml-sycl/
cp -rpv ../llama.cpp/ggml/src/vulkan-shaders/* src/vulkan-shaders/

cp -rpv ../llama.cpp/ggml/include/ggml*.h include/

cp -rpv ../llama.cpp/tests/test-opt.cpp           tests/test-opt.cpp
cp -rpv ../llama.cpp/tests/test-grad0.cpp         tests/test-grad0.cpp
cp -rpv ../llama.cpp/tests/test-quantize-fns.cpp  tests/test-quantize-fns.cpp
cp -rpv ../llama.cpp/tests/test-quantize-perf.cpp tests/test-quantize-perf.cpp
cp -rpv ../llama.cpp/tests/test-backend-ops.cpp   tests/test-backend-ops.cpp

cp -rpv ../llama.cpp/LICENSE                ./LICENSE
cp -rpv ../llama.cpp/scripts/gen-authors.sh ./scripts/gen-authors.sh
