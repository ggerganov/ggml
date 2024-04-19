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
    ggml*.h \
    ggml*.c \
    ggml*.cpp \
    ggml*.m \
    ggml*.metal \
    ggml*.cu \
    ggml-cuda/* \
    whisper.h \
    whisper.cpp \
    examples/common.h \
    examples/common.cpp \
    examples/common-ggml.h \
    examples/common-ggml.cpp \
    examples/grammar-parser.h \
    examples/grammar-parser.cpp \
    examples/main/main.cpp \
    examples/quantize/quantize.cpp \
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
    # ggml.c              -> src/ggml.c
    # ggml-alloc.c        -> src/ggml-alloc.c
    # ggml-backend-impl.h -> src/ggml-backend-impl.h
    # ggml-backend.c      -> src/ggml-backend.c
    # ggml-common.h       -> src/ggml-common.h
    # ggml-cuda/*         -> src/ggml-cuda/
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
    # ggml-vulkan.cpp    -> src/ggml-vulkan.cpp
    # ggml-vulkan.h      -> src/ggml-vulkan.h
    # ggml.h              -> include/ggml/ggml.h
    # ggml-alloc.h        -> include/ggml/ggml-alloc.h
    # ggml-backend.h      -> include/ggml/ggml-backend.h
    #
    # whisper.h           -> examples/whisper/whisper.h
    # whisper.cpp         -> examples/whisper/whisper.cpp
    #
    # examples/common.h              -> examples/common.h
    # examples/common.cpp            -> examples/common.cpp
    # examples/common-ggml.h         -> examples/common-ggml.h
    # examples/common-ggml.cpp       -> examples/common-ggml.cpp
    # examples/grammar-parser.h      -> examples/whisper/grammar-parser.h
    # examples/grammar-parser.cpp    -> examples/whisper/grammar-parser.cpp
    # examples/main/main.cpp         -> examples/whisper/main.cpp
    # examples/quantize/quantize.cpp -> examples/whisper/quantize.cpp
    #
    # LICENSE                -> LICENSE
    # scripts/gen-authors.sh -> scripts/gen-authors.sh

    cat whisper-src.patch | sed \
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
        -e 's/\/whisper\.h/\/examples\/whisper\/whisper.h/g' \
        -e 's/\/whisper\.cpp/\/examples\/whisper\/whisper.cpp/g' \
        -e 's/\/examples\/common\.h/\/examples\/common.h/g' \
        -e 's/\/examples\/common\.cpp/\/examples\/common.cpp/g' \
        -e 's/\/examples\/common-ggml\.h/\/examples\/common-ggml.h/g' \
        -e 's/\/examples\/common-ggml\.cpp/\/examples\/common-ggml.cpp/g' \
        -e 's/\/examples\/grammar-parser\.h/\/examples\/whisper\/grammar-parser.h/g' \
        -e 's/\/examples\/grammar-parser\.cpp/\/examples\/whisper\/grammar-parser.cpp/g' \
        -e 's/\/examples\/main\/main\.cpp/\/examples\/whisper\/main.cpp/g' \
        -e 's/\/examples\/quantize\/quantize\.cpp/\/examples\/whisper\/quantize.cpp/g' \
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
