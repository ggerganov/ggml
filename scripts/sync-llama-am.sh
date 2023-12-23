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
    ggml*.c \
    ggml*.cpp \
    ggml*.m \
    ggml*.metal \
    ggml*.cu \
    tests/tests-opt.cpp \
    tests/tests-grad0.cpp \
    tests/tests-backend-ops.cpp \
    > $SRC_GGML/llama-src.patch

git format-patch $lc --stdout -- \
    ggml*.h \
    > $SRC_GGML/llama-inc.patch

# delete files if empty
if [ ! -s $SRC_GGML/llama-src.patch ]; then
    rm -v $SRC_GGML/llama-src.patch
fi

if [ ! -s $SRC_GGML/llama-inc.patch ]; then
    rm -v $SRC_GGML/llama-inc.patch
fi

cd $SRC_GGML

if [ -f $SRC_GGML/llama-src.patch ]; then
    # replace PR numbers
    # Subject: some text (#1234)
    # Subject: some text (llama/1234)
    cat llama-src.patch | sed -e 's/^Subject: \(.*\) (#\([0-9]*\))/Subject: \1 (llama\/\2)/' > llama-src.patch.tmp
    mv llama-src.patch.tmp llama-src.patch

    git am -p1 --directory src llama-src.patch

    rm -v $SRC_GGML/llama-src.patch
fi

if [ -f $SRC_GGML/llama-inc.patch ]; then
    cat llama-inc.patch | sed -e 's/^Subject: \(.*\) (#\([0-9]*\))/Subject: \1 (llama\/\2)/' > llama-inc.patch.tmp
    mv llama-inc.patch.tmp llama-inc.patch

    git am -p1 --directory include/ggml llama-inc.patch

    rm -v $SRC_GGML/llama-inc.patch
fi

# update last commit
cd $SRC_LLAMA
git log -1 --format=%H > $SRC_GGML/scripts/sync-llama.last

echo "Done"

exit 0
