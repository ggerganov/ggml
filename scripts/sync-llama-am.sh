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
        ggml/src/ggml*.h \
        ggml/src/ggml*.c \
        ggml/src/ggml*.cpp \
        ggml/src/gguf*.cpp \
        ggml/src/ggml-blas/* \
        ggml/src/ggml-cann/* \
        ggml/src/ggml-cpu/* \
        ggml/src/ggml-cuda/* \
        ggml/src/ggml-hip/* \
        ggml/src/ggml-kompute/* \
        ggml/src/ggml-metal/* \
        ggml/src/ggml-musa/* \
        ggml/src/ggml-opencl/* \
        ggml/src/ggml-rpc/* \
        ggml/src/ggml-sycl/* \
        ggml/src/ggml-vulkan/* \
        ggml/include/ggml*.h \
        ggml/include/gguf*.h \
        tests/test-opt.cpp \
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
    #
    # ggml/src/ggml*.c          -> src/ggml*.c
    # ggml/src/ggml*.cpp        -> src/ggml*.cpp
    # ggml/src/ggml*.h          -> src/ggml*.h
    # ggml/src/gguf*.cpp        -> src/gguf*.h
    # ggml/src/ggml-blas/*      -> src/ggml-blas/*
    # ggml/src/ggml-cann/*      -> src/ggml-cann/*
    # ggml/src/ggml-cpu/*       -> src/ggml-cpu/*
    # ggml/src/ggml-cuda/*      -> src/ggml-cuda/*
    # ggml/src/ggml-hip/*       -> src/ggml-hip/*
    # ggml/src/ggml-kompute/*   -> src/ggml-kompute/*
    # ggml/src/ggml-metal/*     -> src/ggml-metal/*
    # ggml/src/ggml-musa/*      -> src/ggml-musa/*
    # ggml/src/ggml-opencl/*    -> src/ggml-opencl/*
    # ggml/src/ggml-rpc/*       -> src/ggml-rpc/*
    # ggml/src/ggml-sycl/*      -> src/ggml-sycl/*
    # ggml/src/ggml-vulkan/*    -> src/ggml-vulkan/*
    #
    # ggml/include/ggml*.h -> include/ggml*.h
    # ggml/include/gguf*.h -> include/gguf*.h
    #
    # tests/test-opt.cpp           -> tests/test-opt.cpp
    # tests/test-quantize-fns.cpp  -> tests/test-quantize-fns.cpp
    # tests/test-quantize-perf.cpp -> tests/test-quantize-perf.cpp
    # tests/test-backend-ops.cpp   -> tests/test-backend-ops.cpp
    #
    # LICENSE                -> LICENSE
    # scripts/gen-authors.sh -> scripts/gen-authors.sh

    cat llama-src.patch | sed -E \
        -e 's/\/ggml\/CMakeLists\.txt/\/CMakeLists.txt/g' \
        -e 's/\/ggml\/src\/CMakeLists\.txt/\/src\/CMakeLists.txt/g' \
        -e 's/\/ggml\/src\/ggml(.*)\.c/\/src\/ggml\1.c/g' \
        -e 's/\/ggml\/src\/ggml(.*)\.cpp/\/src\/ggml\1.cpp/g' \
        -e 's/\/ggml\/src\/ggml(.*)\.h/\/src\/ggml\1.h/g' \
        -e 's/\/ggml\/src\/gguf(.*)\.cpp/\/src\/gguf\1.cpp/g' \
        -e 's/\/ggml\/src\/ggml-blas\//\/src\/ggml-blas\//g' \
        -e 's/\/ggml\/src\/ggml-cann\//\/src\/ggml-cann\//g' \
        -e 's/\/ggml\/src\/ggml-cpu\//\/src\/ggml-cpu\//g' \
        -e 's/\/ggml\/src\/ggml-cuda\//\/src\/ggml-cuda\//g' \
        -e 's/\/ggml\/src\/ggml-hip\//\/src\/ggml-hip\//g' \
        -e 's/\/ggml\/src\/ggml-kompute\//\/src\/ggml-kompute\//g' \
        -e 's/\/ggml\/src\/ggml-metal\//\/src\/ggml-metal\//g' \
        -e 's/\/ggml\/src\/ggml-musa\//\/src\/ggml-musa\//g' \
        -e 's/\/ggml\/src\/ggml-opencl\//\/src\/ggml-opencl\//g' \
        -e 's/\/ggml\/src\/ggml-rpc\//\/src\/ggml-rpc\//g' \
        -e 's/\/ggml\/src\/ggml-sycl\//\/src\/ggml-sycl\//g' \
        -e 's/\/ggml\/src\/ggml-vulkan\//\/src\/ggml-vulkan\//g' \
        -e 's/\/ggml\/include\/ggml(.*)\.h/\/include\/ggml\1.h/g' \
        -e 's/\/ggml\/include\/gguf(.*)\.h/\/include\/gguf\1.h/g' \
        -e 's/\/tests\/test-opt\.cpp/\/tests\/test-opt.cpp/g' \
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
