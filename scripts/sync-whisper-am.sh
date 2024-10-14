#!/bin/bash
#
# Synchronize whisper.cpp changes to ggml
#
# Usage:
#
#   $ cd /path/to/ggml
#   $ ./scripts/sync-whisper-am.sh -skip hash0,hash1,hash2...
#

set -e

sd=$(dirname $0)
cd $sd/../

SRC_GGML=$(pwd)
SRC_WHISPER=$(cd ../whisper.cpp; pwd)

if [ ! -d $SRC_WHISPER ]; then
    echo "whisper.cpp not found at $SRC_WHISPER"
    exit 1
fi

lc=$(cat $SRC_GGML/scripts/sync-whisper.last)
echo "Syncing whisper.cpp changes since commit $lc"

to_skip=""
if [ "$1" == "-skip" ]; then
    to_skip=$2
fi

cd $SRC_WHISPER

git log --oneline $lc..HEAD
git log --oneline $lc..HEAD --reverse | grep -v "(ggml/[0-9]*)" | grep -v "(llama/[0-9]*)" | cut -d' ' -f1 > $SRC_GGML/whisper-commits

if [ ! -s $SRC_GGML/whisper-commits ]; then
    rm -v $SRC_GGML/whisper-commits
    echo "No new commits"
    exit 0
fi

if [ -f $SRC_GGML/whisper-src.patch ]; then
    rm -v $SRC_GGML/whisper-src.patch
fi

while read c; do
    if [ -n "$to_skip" ]; then
        if [[ $to_skip == *"$c"* ]]; then
            echo "Skipping $c"
            continue
        fi
    fi

    git format-patch -k $c~1..$c --stdout -- \
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
        examples/common.h \
        examples/common.cpp \
        examples/common-ggml.h \
        examples/common-ggml.cpp \
        LICENSE \
        scripts/gen-authors.sh \
        >> $SRC_GGML/whisper-src.patch
done < $SRC_GGML/whisper-commits

rm -v $SRC_GGML/whisper-commits

# delete files if empty
if [ ! -s $SRC_GGML/whisper-src.patch ]; then
    rm -v $SRC_GGML/whisper-src.patch
fi

cd $SRC_GGML

if [ -f $SRC_GGML/whisper-src.patch ]; then
    # replace PR numbers
    #
    # Subject: some text (#1234)
    # Subject: some text (whisper/1234)
    cat whisper-src.patch | sed -e 's/^Subject: \(.*\) (#\([0-9]*\))/Subject: \1 (whisper\/\2)/' > whisper-src.patch.tmp
    mv whisper-src.patch.tmp whisper-src.patch

    cat whisper-src.patch | sed -e 's/^\(.*\) (#\([0-9]*\))$/\1 (whisper\/\2)/' > whisper-src.patch.tmp
    mv whisper-src.patch.tmp whisper-src.patch

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
    # examples/common.h        -> examples/common.h
    # examples/common.cpp      -> examples/common.cpp
    # examples/common-ggml.h   -> examples/common-ggml.h
    # examples/common-ggml.cpp -> examples/common-ggml.cpp
    #
    # LICENSE                -> LICENSE
    # scripts/gen-authors.sh -> scripts/gen-authors.sh

    cat whisper-src.patch | sed \
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
        -e 's/\/examples\/common\.h/\/examples\/common.h/g' \
        -e 's/\/examples\/common\.cpp/\/examples\/common.cpp/g' \
        -e 's/\/examples\/common-ggml\.h/\/examples\/common-ggml.h/g' \
        -e 's/\/examples\/common-ggml\.cpp/\/examples\/common-ggml.cpp/g' \
        -e 's/\/LICENSE/\/LICENSE/g' \
        -e 's/\/scripts\/gen-authors\.sh/\/scripts\/gen-authors.sh/g' \
        > whisper-src.patch.tmp
    mv whisper-src.patch.tmp whisper-src.patch

    git am whisper-src.patch

    rm -v $SRC_GGML/whisper-src.patch
fi

# update last commit
cd $SRC_WHISPER
git log -1 --format=%H > $SRC_GGML/scripts/sync-whisper.last

echo "Done"

exit 0
