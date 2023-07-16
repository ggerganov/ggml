#/bin/bash

sd=`dirname $0`
cd $sd/../

SRC=`pwd`
OUT=$1

## helpers

function gg_printf {
    printf -- "$@" >> $OUT/README.md
}

function gg_run {
    ci=$1

    gg_run_$ci
    cur=$?
    echo "$cur" > $OUT/$ci.exit

    gg_sum_$ci

    return $cur
}

## ci

function gg_run_ci_0 {
    cd $SRC

    rm -rf build-ci && mkdir build-ci && cd build-ci

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Debug ..     ) 2>&1 | tee $OUT/ci-0-cmake.log
    (time make -j4                              ) 2>&1 | tee $OUT/ci-0-make.log
    (time ctest --output-on-failure -E test-opt ) 2>&1 | tee $OUT/ci-0-ctest.log

    set +e
}

function gg_sum_ci_0 {
    gg_printf '### ci-0\n\n'

    gg_printf '- status: ' "$(cat $OUT/ci-0.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/ci-0-ctest.log)"
    gg_printf '```\n'
    gg_printf '\n'
}

function gg_run_ci_1 {
    cd $SRC

    rm -rf build-ci && mkdir build-ci && cd build-ci

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Release ..   ) 2>&1 | tee $OUT/ci-1-cmake.log
    (time make -j4                              ) 2>&1 | tee $OUT/ci-1-make.log
    (time ctest --output-on-failure -E test-opt ) 2>&1 | tee $OUT/ci-1-ctest.log

    set +e
}

function gg_sum_ci_1 {
    gg_printf '### ci-1\n\n'

    gg_printf '- status: ' "$(cat $OUT/ci-1.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/ci-1-ctest.log)"
    gg_printf '```\n'
}

## main

ret=0

ret=$(($ret + $(gg_run ci_0)))
ret=$(($ret + $(gg_run ci_1)))

exit $ret
