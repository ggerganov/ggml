#!/bin/bash
#
# Synchronize llama.cpp changes to ggml
#
# Usage:
#
#   $ cd /path/to/ggml
#   $ ./scripts/sync-llama-am.sh -skip hash0,hash1,hash2...
#

set -e

sd=$(dirname $0)
cd $sd/../

SRC_GGML=$(pwd)
SRC_LLAMA=$(cd ../llama.cpp; pwd)

if [ ! -d $SRC_LLAMA ]; then
    echo "llama.cpp not found at $SRC_LLAMA"
    exit 1
fi

lc=$(cat $SRC_GGML/scripts/sync-llama.last)
echo "Syncing llama.cpp changes since commit $lc"

to_skip=""
if [ "$1" == "-skip" ]; then
    to_skip=$2
fi

cd $SRC_LLAMA

git log --oneline $lc..HEAD
git log --oneline $lc..HEAD --reverse | grep -v "(ggml/[0-9]*)" | grep -v "(whisper/[0-9]*)" | cut -d' ' -f1 > $SRC_GGML/llama-commits

if [ ! -s $SRC_GGML/llama-commits ]; then
    rm -v $SRC_GGML/llama-commits
    echo "No new commits"
    exit 0
fi

if [ -f $SRC_GGML/llama-src.patch ]; then
    rm -v $SRC_GGML/llama-src.patch
fi

while read c; do
    if [ -n "$to_skip" ]; then
        if [[ $to_skip == *"$c"* ]]; then
            echo "Skipping $c"
            continue
        fi
    fi

    git format-patch -k $c~1..$c --stdout -- \
        ggml*.h \
        ggml*.c \
        ggml*.cpp \
        ggml*.m \
        ggml*.metal \
        ggml*.cu \
        ggml-cuda/* \
        tests/test-opt.cpp \
        tests/test-grad0.cpp \
        tests/test-quantize-fns.cpp \
        tests/test-quantize-perf.cpp \
        tests/test-backend-ops.cpp \
        LICENSE \
        scripts/gen-authors.sh \
        >> $SRC_GGML/llama-src.patch
done < $SRC_GGML/llama-commits

rm -v $SRC_GGML/llama-commits

# delete files if empty
if [ ! -s $SRC_GGML/llama-src.patch ]; then
    rm -v $SRC_GGML/llama-src.patch
fi

cd $SRC_GGML

if [ -f $SRC_GGML/llama-src.patch ]; then
    # replace PR numbers
    #
    # Subject: some text (#1234)
    # Subject: some text (llama/1234)
    cat llama-src.patch | sed -e 's/^Subject: \(.*\) (#\([0-9]*\))/Subject: \1 (llama\/\2)/' > llama-src.patch.tmp
    mv llama-src.patch.tmp llama-src.patch

    cat llama-src.patch | sed -e 's/^\(.*\) (#\([0-9]*\))$/\1 (llama\/\2)/' > llama-src.patch.tmp
    mv llama-src.patch.tmp llama-src.patch

    # replace filenames:
    #
    # ggml.c              -> src/ggml.c
    # ggml-alloc.c        -> src/ggml-alloc.c
    # ggml-backend-impl.h -> src/ggml-backend-impl.h
    # ggml-backend.c      -> src/ggml-backend.c
    # ggml-common.h       -> src/ggml-common.h
    # ggml-cuda/*         -> src/ggml-cuda/*
    # ggml-cuda.cu        -> src/ggml-cuda.cu
    # ggml-cuda.h         -> src/ggml-cuda.h
    # ggml-impl.h         -> src/ggml-impl.h
    # ggml-kompute.cpp    -> src/ggml-kompute.cpp
    # ggml-kompute.h      -> src/ggml-kompute.h
    # ggml-metal.h        -> src/ggml-metal.h
    # ggml-metal.m        -> src/ggml-metal.m
    # ggml-mpi.h          -> src/ggml-mpi.h
    # ggml-mpi.c          -> src/ggml-mpi.c
    # ggml-opencl.cpp     -> src/ggml-opencl.cpp
    # ggml-opencl.h       -> src/ggml-opencl.h
    # ggml-quants.c       -> src/ggml-quants.c
    # ggml-quants.h       -> src/ggml-quants.h
    # ggml-sycl.cpp       -> src/ggml-sycl.cpp
    # ggml-sycl.h         -> src/ggml-sycl.h
    # ggml-vulkan.cpp     -> src/ggml-vulkan.cpp
    # ggml-vulkan.h       -> src/ggml-vulkan.h
    # ggml.h              -> include/ggml/ggml.h
    # ggml-alloc.h        -> include/ggml/ggml-alloc.h
    # ggml-backend.h      -> include/ggml/ggml-backend.h
    #
    # tests/test-opt.cpp           -> tests/test-opt.cpp
    # tests/test-grad0.cpp         -> tests/test-grad0.cpp
    # tests/test-quantize-fns.cpp  -> tests/test-quantize-fns.cpp
    # tests/test-quantize-perf.cpp -> tests/test-quantize-perf.cpp
    # tests/test-backend-ops.cpp   -> tests/test-backend-ops.cpp
    #
    # LICENSE                -> LICENSE
    # scripts/gen-authors.sh -> scripts/gen-authors.sh

    cat llama-src.patch | sed \
        -e 's/\/ggml\.c/\/src\/ggml.c/g' \
        -e 's/\/ggml-alloc\.c/\/src\/ggml-alloc.c/g' \
        -e 's/\/ggml-backend-impl\.h/\/src\/ggml-backend-impl.h/g' \
        -e 's/\/ggml-backend\.c/\/src\/ggml-backend.c/g' \
        -e 's/\/ggml-common\.h/\/src\/ggml-common.h/g' \
        -e 's/\/ggml-cuda\//\/src\/ggml-cuda\//g' \
        -e 's/\/ggml-cuda\.cu/\/src\/ggml-cuda.cu/g' \
        -e 's/\/ggml-cuda\.h/\/src\/ggml-cuda.h/g' \
        -e 's/\/ggml-impl\.h/\/src\/ggml-impl.h/g' \
        -e 's/\/ggml-kompute\.cpp/\/src\/ggml-kompute.cpp/g' \
        -e 's/\/ggml-kompute\.h/\/src\/ggml-kompute.h/g' \
        -e 's/\/ggml-metal\.h/\/src\/ggml-metal.h/g' \
        -e 's/\/ggml-metal\.m/\/src\/ggml-metal.m/g' \
        -e 's/\/ggml-mpi\.h/\/src\/ggml-mpi.h/g' \
        -e 's/\/ggml-mpi\.c/\/src\/ggml-mpi.c/g' \
        -e 's/\/ggml-opencl\.cpp/\/src\/ggml-opencl.cpp/g' \
        -e 's/\/ggml-opencl\.h/\/src\/ggml-opencl.h/g' \
        -e 's/\/ggml-quants\.c/\/src\/ggml-quants.c/g' \
        -e 's/\/ggml-quants\.h/\/src\/ggml-quants.h/g' \
        -e 's/\/ggml-sycl\.cpp/\/src\/ggml-sycl.cpp/g' \
        -e 's/\/ggml-sycl\.h/\/src\/ggml-sycl.h/g' \
        -e 's/\/ggml-vulkan\.cpp/\/src\/ggml-vulkan.cpp/g' \
        -e 's/\/ggml-vulkan\.h/\/src\/ggml-vulkan.h/g' \
        -e 's/\/ggml\.h/\/include\/ggml\/ggml.h/g' \
        -e 's/\/ggml-alloc\.h/\/include\/ggml\/ggml-alloc.h/g' \
        -e 's/\/ggml-backend\.h/\/include\/ggml\/ggml-backend.h/g' \
        -e 's/\/tests\/test-opt\.cpp/\/tests\/test-opt.cpp/g' \
        -e 's/\/tests\/test-grad0\.cpp/\/tests\/test-grad0.cpp/g' \
        -e 's/\/tests\/test-quantize-fns\.cpp/\/tests\/test-quantize-fns.cpp/g' \
        -e 's/\/tests\/test-quantize-perf\.cpp/\/tests\/test-quantize-perf.cpp/g' \
        -e 's/\/tests\/test-backend-ops\.cpp/\/tests\/test-backend-ops.cpp/g' \
        -e 's/\/LICENSE/\/LICENSE/g' \
        -e 's/\/scripts\/gen-authors\.sh/\/scripts\/gen-authors.sh/g' \
        > llama-src.patch.tmp
    mv llama-src.patch.tmp llama-src.patch

    git am llama-src.patch

    rm -v $SRC_GGML/llama-src.patch
fi

# update last commit
cd $SRC_LLAMA
git log -1 --format=%H > $SRC_GGML/scripts/sync-llama.last

echo "Done"

exit 0
