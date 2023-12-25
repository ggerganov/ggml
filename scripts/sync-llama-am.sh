#!/bin/bash
#
# Synchronize llama.cpp changes to ggml
#
# Usage:
#
#   $ cd /path/to/ggml
#   $ ./scripts/sync-llama-am.sh
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

cd $SRC_LLAMA

git log --oneline $lc..HEAD

git format-patch $lc --stdout -- \
    ggml*.h \
    ggml*.c \
    ggml*.cpp \
    ggml*.m \
    ggml*.metal \
    ggml*.cu \
    tests/test-opt.cpp \
    tests/test-grad0.cpp \
    tests/test-quantize-fns.cpp \
    tests/test-quantize-perf.cpp \
    tests/test-backend-ops.cpp \
    > $SRC_GGML/llama-src.patch

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
    # ggml-cuda.cu        -> src/ggml-cuda.cu
    # ggml-cuda.h         -> src/ggml-cuda.h
    # ggml-impl.h         -> src/ggml-impl.h
    # ggml-metal.h        -> src/ggml-metal.h
    # ggml-metal.m        -> src/ggml-metal.m
    # ggml-metal.metal    -> src/ggml-metal.metal
    # ggml-mpi.h          -> src/ggml-mpi.h
    # ggml-mpi.c          -> src/ggml-mpi.c
    # ggml-opencl.cpp     -> src/ggml-opencl.cpp
    # ggml-opencl.h       -> src/ggml-opencl.h
    # ggml-quants.c       -> src/ggml-quants.c
    # ggml-quants.h       -> src/ggml-quants.h
    # ggml.h              -> include/ggml/ggml.h
    # ggml-alloc.h        -> include/ggml/ggml-alloc.h
    # ggml-backend.h      -> include/ggml/ggml-backend.h
    #
    # tests/test-opt.cpp           -> tests/test-opt.cpp
    # tests/test-grad0.cpp         -> tests/test-grad0.cpp
    # tests/test-quantize-fns.cpp  -> tests/test-quantize-fns.cpp
    # tests/test-quantize-perf.cpp -> tests/test-quantize-perf.cpp
    # tests/test-backend-ops.cpp   -> tests/test-backend-ops.cpp

    cat llama-src.patch | sed \
        -e 's/\/ggml\.c/\/src\/ggml.c/' \
        -e 's/\/ggml-alloc\.c/\/src\/ggml-alloc.c/' \
        -e 's/\/ggml-backend-impl\.h/\/src\/ggml-backend-impl.h/' \
        -e 's/\/ggml-backend\.c/\/src\/ggml-backend.c/' \
        -e 's/\/ggml-cuda\.cu/\/src\/ggml-cuda.cu/' \
        -e 's/\/ggml-cuda\.h/\/src\/ggml-cuda.h/' \
        -e 's/\/ggml-impl\.h/\/src\/ggml-impl.h/' \
        -e 's/\/ggml-metal\.h/\/src\/ggml-metal.h/' \
        -e 's/\/ggml-metal\.m/\/src\/ggml-metal.m/' \
        -e 's/\/ggml-metal\.metal/\/src\/ggml-metal.metal/' \
        -e 's/\/ggml-mpi\.h/\/src\/ggml-mpi.h/' \
        -e 's/\/ggml-mpi\.c/\/src\/ggml-mpi.c/' \
        -e 's/\/ggml-opencl\.cpp/\/src\/ggml-opencl.cpp/' \
        -e 's/\/ggml-opencl\.h/\/src\/ggml-opencl.h/' \
        -e 's/\/ggml-quants\.c/\/src\/ggml-quants.c/' \
        -e 's/\/ggml-quants\.h/\/src\/ggml-quants.h/' \
        -e 's/\/ggml\.h/\/include\/ggml\/ggml.h/' \
        -e 's/\/ggml-alloc\.h/\/include\/ggml\/ggml-alloc.h/' \
        -e 's/\/ggml-backend\.h/\/include\/ggml\/ggml-backend.h/' \
        -e 's/\/tests\/test-opt\.cpp/\/tests\/test-opt.cpp/' \
        -e 's/\/tests\/test-grad0\.cpp/\/tests\/test-grad0.cpp/' \
        -e 's/\/tests\/test-quantize-fns\.cpp/\/tests\/test-quantize-fns.cpp/' \
        -e 's/\/tests\/test-quantize-perf\.cpp/\/tests\/test-quantize-perf.cpp/' \
        -e 's/\/tests\/test-backend-ops\.cpp/\/tests\/test-backend-ops.cpp/' \
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
