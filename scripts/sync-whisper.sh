#!/bin/bash

cp -rpv ../whisper.cpp/ggml.c                         src/ggml.c
cp -rpv ../whisper.cpp/ggml-impl.h                    src/ggml-impl.h
cp -rpv ../whisper.cpp/ggml-alloc.c                   src/ggml-alloc.c
cp -rpv ../whisper.cpp/ggml-backend-impl.h            src/ggml-backend-impl.h
cp -rpv ../whisper.cpp/ggml-backend.c                 src/ggml-backend.c
cp -rpv ../whisper.cpp/ggml-cuda.cu                   src/ggml-cuda.cu
cp -rpv ../whisper.cpp/ggml-cuda.h                    src/ggml-cuda.h
cp -rpv ../whisper.cpp/ggml-kompute.cpp               src/ggml-kompute.cpp
cp -rpv ../whisper.cpp/ggml-kompute.h                 src/ggml-kompute.h
cp -rpv ../whisper.cpp/ggml-metal.h                   src/ggml-metal.h
cp -rpv ../whisper.cpp/ggml-metal.m                   src/ggml-metal.m
cp -rpv ../whisper.cpp/ggml-metal.metal               src/ggml-metal.metal
#cp -rpv ../whisper.cpp/ggml-mpi.h                     src/ggml-mpi.h
#cp -rpv ../whisper.cpp/ggml-mpi.m                     src/ggml-mpi.m
cp -rpv ../whisper.cpp/ggml-opencl.cpp                src/ggml-opencl.cpp
cp -rpv ../whisper.cpp/ggml-opencl.h                  src/ggml-opencl.h
cp -rpv ../whisper.cpp/ggml-quants.c                  src/ggml-quants.c
cp -rpv ../whisper.cpp/ggml-quants.h                  src/ggml-quants.h
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

cp -rpv ../whisper.cpp/whisper.h                      examples/whisper/whisper.h
cp -rpv ../whisper.cpp/whisper.cpp                    examples/whisper/whisper.cpp
cp -rpv ../whisper.cpp/examples/main/main.cpp         examples/whisper/main.cpp
cp -rpv ../whisper.cpp/examples/quantize/quantize.cpp examples/whisper/quantize.cpp
