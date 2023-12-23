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
git format-patch $lc --stdout -- ggml* > $SRC_GGML/llama-am.patch

cd $SRC_GGML

# replace PR numbers
# Subject: some text (#1234)
# Subject: some text (llama/1234)
cat llama-am.patch | sed -e 's/^Subject: \(.*\) (#\([0-9]*\))/Subject: \1 (llama\/\2)/' > llama-am.patch.tmp
mv llama-am.patch.tmp llama-am.patch

git am -p1 --directory src llama-am.patch

# update last commit
cd ../llama.cpp
git log -1 --format=%H > $SRC_GGML/scripts/sync-llama.last

# clean up
rm -v $SRC_GGML/llama-am.patch

echo "Done"

exit 0
