#!/bin/bash

cp -rpv ../whisper.cpp/ggml/CMakeLists.txt       CMakeLists.txt
cp -rpv ../whisper.cpp/ggml/src/CMakeLists.txt   src/CMakeLists.txt
cp -rpv ../whisper.cpp/ggml/cmake/FindSIMD.cmake cmake/FindSIMD.cmake

cp -rpv ../whisper.cpp/ggml/src/ggml*.c          src/
cp -rpv ../whisper.cpp/ggml/src/ggml*.cpp        src/
cp -rpv ../whisper.cpp/ggml/src/ggml*.h          src/
cp -rpv ../whisper.cpp/ggml/src/ggml*.cu         src/
cp -rpv ../whisper.cpp/ggml/src/ggml*.m          src/
cp -rpv ../whisper.cpp/ggml/src/ggml-amx/*       src/ggml-amx/
cp -rpv ../whisper.cpp/ggml/src/ggml-cann/*      src/ggml-cann/
cp -rpv ../whisper.cpp/ggml/src/ggml-cuda/*      src/ggml-cuda/
cp -rpv ../whisper.cpp/ggml/src/ggml-sycl/*      src/ggml-sycl/
cp -rpv ../whisper.cpp/ggml/src/vulkan-shaders/* src/vulkan-shaders/

cp -rpv ../whisper.cpp/ggml/include/ggml*.h include/

cp -rpv ../whisper.cpp/examples/common.h        examples/common.h
cp -rpv ../whisper.cpp/examples/common.cpp      examples/common.cpp
cp -rpv ../whisper.cpp/examples/common-ggml.h   examples/common-ggml.h
cp -rpv ../whisper.cpp/examples/common-ggml.cpp examples/common-ggml.cpp

cp -rpv ../whisper.cpp/LICENSE                ./LICENSE
cp -rpv ../whisper.cpp/scripts/gen-authors.sh ./scripts/gen-authors.sh
