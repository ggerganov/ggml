#!/bin/bash

cp -rpv ../whisper.cpp/ggml.c                         src/ggml.c
cp -rpv ../whisper.cpp/ggml-alloc.c                   src/ggml-alloc.c
cp -rpv ../whisper.cpp/ggml-cuda.h                    src/ggml-cuda.h
cp -rpv ../whisper.cpp/ggml-cuda.cu                   src/ggml-cuda.cu
cp -rpv ../whisper.cpp/ggml-opencl.h                  src/ggml-opencl.h
cp -rpv ../whisper.cpp/ggml-opencl.cpp                src/ggml-opencl.cpp
cp -rpv ../whisper.cpp/ggml-metal.h                   src/ggml-metal.h
cp -rpv ../whisper.cpp/ggml-metal.m                   src/ggml-metal.m
cp -rpv ../whisper.cpp/ggml-metal.metal               src/ggml-metal.metal
cp -rpv ../whisper.cpp/ggml.h                         include/ggml/ggml.h
cp -rpv ../whisper.cpp/ggml-alloc.h                   include/ggml/ggml-alloc.h
cp -rpv ../whisper.cpp/examples/common.h              examples/common.h
cp -rpv ../whisper.cpp/examples/common.cpp            examples/common.cpp
cp -rpv ../whisper.cpp/examples/common-ggml.h         examples/common-ggml.h
cp -rpv ../whisper.cpp/examples/common-ggml.cpp       examples/common-ggml.cpp
cp -rpv ../whisper.cpp/whisper.h                      examples/whisper/whisper.h
cp -rpv ../whisper.cpp/whisper.cpp                    examples/whisper/whisper.cpp
cp -rpv ../whisper.cpp/examples/main/main.cpp         examples/whisper/main.cpp
cp -rpv ../whisper.cpp/examples/quantize/quantize.cpp examples/whisper/quantize.cpp
