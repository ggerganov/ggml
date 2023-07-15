#/bin/bash

sd=`dirname $0`
cd $sd/../

SRC=`pwd`
OUT=$1

## ci

function gg_ci_0() {
    cd $SRC

    mkdir build-ci-0
    cd build-ci-0

    set -e

    time cmake -DCMAKE_BUILD_TYPE=Debug .. | tee $OUT/ci-0-cmake.log
    time make -j4 | tee $OUT/ci-0-make.log

    time ctest -E test-opt | tee $OUT/ci-0-ctest.log

    set +e
}

function gg_ci_1() {
    exit 1
}

## main

ret=0

gg_ci_0 | tee $OUT/ci-0.log
cur=$?
echo "$cur" > $OUT/ci-0.exit
ret=$(($ret + $cur))

gg_ci_1 | tee $OUT/ci-1.log
cur=$?
echo "$cur" > $OUT/ci-1.exit
ret=$(($ret + $cur))

exit $ret
