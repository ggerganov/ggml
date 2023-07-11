#!/bin/bash

cp -rpv ../llama.cpp/ggml.c           src/ggml.c
cp -rpv ../llama.cpp/ggml-cuda.h      src/ggml-cuda.h
cp -rpv ../llama.cpp/ggml-cuda.cu     src/ggml-cuda.cu
cp -rpv ../llama.cpp/ggml-opencl.h    src/ggml-opencl.h
cp -rpv ../llama.cpp/ggml-opencl.cpp  src/ggml-opencl.cpp
cp -rpv ../llama.cpp/ggml-metal.h     src/ggml-metal.h
cp -rpv ../llama.cpp/ggml-metal.m     src/ggml-metal.m
cp -rpv ../llama.cpp/ggml-metal.metal src/ggml-metal.metal
cp -rpv ../llama.cpp/ggml.h           include/ggml/ggml.h

cp -rpv ../llama.cpp/tests/test-opt.c             tests/test-opt.c
cp -rpv ../llama.cpp/tests/test-grad0.c           tests/test-grad0.c
cp -rpv ../llama.cpp/tests/test-quantize-fns.cpp  tests/test-quantize-fns.cpp
cp -rpv ../llama.cpp/tests/test-quantize-perf.cpp tests/test-quantize-perf.cpp
