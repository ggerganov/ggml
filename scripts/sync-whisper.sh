#!/bin/bash

cp -rpv ../whisper.cpp/ggml.c                         src/ggml.c
cp -rpv ../whisper.cpp/ggml-impl.h                    src/ggml-impl.h
cp -rpv ../whisper.cpp/ggml-alloc.c                   src/ggml-alloc.c
cp -rpv ../whisper.cpp/ggml-backend-impl.h            src/ggml-backend-impl.h
cp -rpv ../whisper.cpp/ggml-backend.c                 src/ggml-backend.c
cp -rpv ../whisper.cpp/ggml-blas.cpp                  src/ggml-blas.cpp
cp -rpv ../whisper.cpp/ggml-blas.h                    src/ggml-blas.h
cp -rpv ../whisper.cpp/ggml-common.h                  src/ggml-common.h
cp -rpv ../whisper.cpp/ggml-cuda/*                    src/ggml-cuda/
cp -rpv ../whisper.cpp/ggml-cuda.cu                   src/ggml-cuda.cu
cp -rpv ../whisper.cpp/ggml-cuda.h                    src/ggml-cuda.h
cp -rpv ../whisper.cpp/ggml-kompute.cpp               src/ggml-kompute.cpp
cp -rpv ../whisper.cpp/ggml-kompute.h                 src/ggml-kompute.h
cp -rpv ../whisper.cpp/ggml-metal.h                   src/ggml-metal.h
cp -rpv ../whisper.cpp/ggml-metal.m                   src/ggml-metal.m
cp -rpv ../whisper.cpp/ggml-metal.metal               src/ggml-metal.metal
cp -rpv ../whisper.cpp/ggml-quants.c                  src/ggml-quants.c
cp -rpv ../whisper.cpp/ggml-quants.h                  src/ggml-quants.h
cp -rpv ../whisper.cpp/ggml-rpc.cpp                   src/ggml-rpc.cpp
cp -rpv ../whisper.cpp/ggml-rpc.h                     src/ggml-rpc.h
cp -rpv ../whisper.cpp/ggml-sycl/*                    src/ggml-sycl/
cp -rpv ../whisper.cpp/ggml-sycl.cpp                  src/ggml-sycl.cpp
cp -rpv ../whisper.cpp/ggml-sycl.h                    src/ggml-sycl.h
cp -rpv ../whisper.cpp/ggml-vulkan.cpp                src/ggml-vulkan.cpp
cp -rpv ../whisper.cpp/ggml-vulkan.h                  src/ggml-vulkan.h

cp -rpv ../whisper.cpp/ggml.h                         include/ggml/ggml.h
cp -rpv ../whisper.cpp/ggml-alloc.h                   include/ggml/ggml-alloc.h
cp -rpv ../whisper.cpp/ggml-backend.h                 include/ggml/ggml-backend.h

cp -rpv ../whisper.cpp/examples/common.h              examples/common.h
cp -rpv ../whisper.cpp/examples/common.cpp            examples/common.cpp
cp -rpv ../whisper.cpp/examples/common-ggml.h         examples/common-ggml.h
cp -rpv ../whisper.cpp/examples/common-ggml.cpp       examples/common-ggml.cpp

cp -rpv ../whisper.cpp/LICENSE                        ./LICENSE
cp -rpv ../whisper.cpp/scripts/gen-authors.sh         ./scripts/gen-authors.sh
