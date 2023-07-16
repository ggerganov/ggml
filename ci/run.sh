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

    set -o pipefail

    gg_run_$ci | tee $OUT/$ci.log
    cur=$?
    echo "$cur" > $OUT/$ci.exit

    set +o pipefail

    gg_sum_$ci

    ret=$((ret | cur))
}

## ci

function gg_run_ci_0 {
    cd $SRC

    rm -rf build-ci && mkdir build-ci && cd build-ci

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Debug ..     ) 2>&1 | tee $OUT/${ci}-cmake.log
    (time make -j4                              ) 2>&1 | tee $OUT/${ci}-make.log
    (time ctest --output-on-failure -E test-opt ) 2>&1 | tee $OUT/${ci}-ctest.log

    set +e
}

function gg_sum_ci_0 {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
    gg_printf '\n'
}

function gg_run_ci_1 {
    cd $SRC

    rm -rf build-ci && mkdir build-ci && cd build-ci

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Release ..   ) 2>&1 | tee $OUT/${ci}-cmake.log
    (time make -j4                              ) 2>&1 | tee $OUT/${ci}-make.log
    (time ctest --output-on-failure -E test-opt ) 2>&1 | tee $OUT/${ci}-ctest.log

    set +e
}

function gg_sum_ci_1 {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
}

## main

ret=0

gg_run ci_0
gg_run ci_1

exit $ret
