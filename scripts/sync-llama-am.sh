#!/bin/bash
#
# Synchronize llama.cpp changes to ggml
#
# Usage:
#
#   $ cd /path/to/ggml
#   $ ./scripts/sync-llama-am.sh -skip hash0,hash1,hash2... -C 3
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

# context for git patches in number of lines
ctx="8"

while [ "$1" != "" ]; do
    case $1 in
        -skip )
            shift
            to_skip=$1
            ;;
        -C )
            shift
            ctx=$1
            ;;
    esac
    shift
done

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

    git format-patch -U${ctx} -k $c~1..$c --stdout -- \
        ggml/CMakeLists.txt \
        ggml/src/CMakeLists.txt \
        ggml/cmake/FindSIMD.cmake \
        ggml/src/ggml*.h \
        ggml/src/ggml*.c \
        ggml/src/ggml*.cpp \
        ggml/src/ggml*.m \
        ggml/src/ggml*.metal \
        ggml/src/ggml*.cu \
        ggml/src/ggml-cann/* \
        ggml/src/ggml-cuda/* \
        ggml/src/ggml-sycl/* \
        ggml/src/vulkan-shaders/* \
        ggml/include/ggml*.h \
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
    # ggml/CMakelists.txt       -> CMakeLists.txt
    # ggml/src/CMakelists.txt   -> src/CMakeLists.txt
    # ggml/cmake/FindSIMD.cmake -> cmake/FindSIMD.cmake
    #
    # ggml/src/ggml.c              -> src/ggml.c
    # ggml/src/ggml-aarch64.c      -> src/ggml-aarch64.c
    # ggml/src/ggml-aarch64.h      -> src/ggml-aarch64.h
    # ggml/src/ggml-alloc.c        -> src/ggml-alloc.c
    # ggml/src/ggml-backend-impl.h -> src/ggml-backend-impl.h
    # ggml/src/ggml-backend.cpp    -> src/ggml-backend.cpp
    # ggml/src/ggml-blas.cpp       -> src/ggml-blas.cpp
    # ggml/src/ggml-cann/*         -> src/ggml-cann/*
    # ggml/src/ggml-cann.cpp       -> src/ggml-cann.cpp
    # ggml/src/ggml-common.h       -> src/ggml-common.h
    # ggml/src/ggml-cuda/*         -> src/ggml-cuda/*
    # ggml/src/ggml-cuda.cu        -> src/ggml-cuda.cu
    # ggml/src/ggml-impl.h         -> src/ggml-impl.h
    # ggml/src/ggml-kompute.cpp    -> src/ggml-kompute.cpp
    # ggml/src/ggml-metal.m        -> src/ggml-metal.m
    # ggml/src/ggml-quants.c       -> src/ggml-quants.c
    # ggml/src/ggml-quants.h       -> src/ggml-quants.h
    # ggml/src/ggml-rpc.cpp        -> src/ggml-rpc.cpp
    # ggml/src/ggml-sycl/*         -> src/ggml-sycl/*
    # ggml/src/ggml-sycl.cpp       -> src/ggml-sycl.cpp
    # ggml/src/ggml-vulkan.cpp     -> src/ggml-vulkan.cpp
    # ggml/src/vulkan-shaders/*    -> src/vulkan-shaders/*
    #
    # ggml/include/ggml.h         -> include/ggml.h
    # ggml/include/ggml-alloc.h   -> include/ggml-alloc.h
    # ggml/include/ggml-backend.h -> include/ggml-backend.h
    # ggml/include/ggml-blas.h    -> include/ggml-blas.h
    # ggml/include/ggml-cann.h    -> include/ggml-cann.h
    # ggml/include/ggml-cuda.h    -> include/ggml-cuda.h
    # ggml/include/ggml-kompute.h -> include/ggml-kompute.h
    # ggml/include/ggml-metal.h   -> include/ggml-metal.h
    # ggml/include/ggml-rpc.h     -> include/ggml-rpc.h
    # ggml/include/ggml-sycl.h    -> include/ggml-sycl.h
    # ggml/include/ggml-vulkan.h  -> include/ggml-vulkan.h
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
        -e 's/\/ggml\/CMakeLists\.txt/\/CMakeLists.txt/g' \
        -e 's/\/ggml\/src\/CMakeLists\.txt/\/src\/CMakeLists.txt/g' \
        -e 's/\/ggml\/cmake\/FindSIMD\.cmake/\/cmake\/FindSIMD.cmake/g' \
        -e 's/\/ggml\/src\/ggml\.c/\/src\/ggml.c/g' \
        -e 's/\/ggml\/src\/ggml-aarch64\.c/\/src\/ggml-aarch64.c/g' \
        -e 's/\/ggml\/src\/ggml-aarch64\.h/\/src\/ggml-aarch64.h/g' \
        -e 's/\/ggml\/src\/ggml-alloc\.c/\/src\/ggml-alloc.c/g' \
        -e 's/\/ggml\/src\/ggml-backend-impl\.h/\/src\/ggml-backend-impl.h/g' \
        -e 's/\/ggml\/src\/ggml-backend\.cpp/\/src\/ggml-backend.cpp/g' \
        -e 's/\/ggml\/src\/ggml-blas\.cpp/\/src\/ggml-blas.cpp/g' \
        -e 's/\/ggml\/src\/ggml-cann\//\/src\/ggml-cann\//g' \
        -e 's/\/ggml\/src\/ggml-cann\.cpp/\/src\/ggml-cann.cpp/g' \
        -e 's/\/ggml\/src\/ggml-common\.h/\/src\/ggml-common.h/g' \
        -e 's/\/ggml\/src\/ggml-cuda\//\/src\/ggml-cuda\//g' \
        -e 's/\/ggml\/src\/ggml-cuda\.cu/\/src\/ggml-cuda.cu/g' \
        -e 's/\/ggml\/src\/ggml-impl\.h/\/src\/ggml-impl.h/g' \
        -e 's/\/ggml\/src\/ggml-kompute\.cpp/\/src\/ggml-kompute.cpp/g' \
        -e 's/\/ggml\/src\/ggml-metal\.m/\/src\/ggml-metal.m/g' \
        -e 's/\/ggml\/src\/ggml-quants\.c/\/src\/ggml-quants.c/g' \
        -e 's/\/ggml\/src\/ggml-quants\.h/\/src\/ggml-quants.h/g' \
        -e 's/\/ggml\/src\/ggml-rpc\.cpp/\/src\/ggml-rpc.cpp/g' \
        -e 's/\/ggml\/src\/ggml-sycl\//\/src\/ggml-sycl\//g' \
        -e 's/\/ggml\/src\/ggml-sycl\.cpp/\/src\/ggml-sycl.cpp/g' \
        -e 's/\/ggml\/src\/ggml-vulkan\.cpp/\/src\/ggml-vulkan.cpp/g' \
        -e 's/\/ggml\/src\/vulkan-shaders\//\/src\/vulkan-shaders\//g' \
        -e 's/\/ggml\/include\/ggml\.h/\/include\/ggml.h/g' \
        -e 's/\/ggml\/include\/ggml-alloc\.h/\/include\/ggml-alloc.h/g' \
        -e 's/\/ggml\/include\/ggml-backend\.h/\/include\/ggml-backend.h/g' \
        -e 's/\/ggml\/include\/ggml-blas\.h/\/include\/ggml-blas.h/g' \
        -e 's/\/ggml\/include\/ggml-cann\.h/\/include\/ggml-cann.h/g' \
        -e 's/\/ggml\/include\/ggml-cuda\.h/\/include\/ggml-cuda.h/g' \
        -e 's/\/ggml\/include\/ggml-kompute\.h/\/include\/ggml-kompute.h/g' \
        -e 's/\/ggml\/include\/ggml-metal\.h/\/include\/ggml-metal.h/g' \
        -e 's/\/ggml\/include\/ggml-rpc\.h/\/include\/ggml-rpc.h/g' \
        -e 's/\/ggml\/include\/ggml-sycl\.h/\/include\/ggml-sycl.h/g' \
        -e 's/\/ggml\/include\/ggml-vulkan\.h/\/include\/ggml-vulkan.h/g' \
        -e 's/\/tests\/test-opt\.cpp/\/tests\/test-opt.cpp/g' \
        -e 's/\/tests\/test-grad0\.cpp/\/tests\/test-grad0.cpp/g' \
        -e 's/\/tests\/test-quantize-fns\.cpp/\/tests\/test-quantize-fns.cpp/g' \
        -e 's/\/tests\/test-quantize-perf\.cpp/\/tests\/test-quantize-perf.cpp/g' \
        -e 's/\/tests\/test-backend-ops\.cpp/\/tests\/test-backend-ops.cpp/g' \
        -e 's/\/LICENSE/\/LICENSE/g' \
        -e 's/\/scripts\/gen-authors\.sh/\/scripts\/gen-authors.sh/g' \
        > llama-src.patch.tmp
    mv llama-src.patch.tmp llama-src.patch

    git am -C${ctx} llama-src.patch

    rm -v $SRC_GGML/llama-src.patch
fi

# update last commit
cd $SRC_LLAMA
git log -1 --format=%H > $SRC_GGML/scripts/sync-llama.last

echo "Done"

exit 0
