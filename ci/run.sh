#/bin/bash

sd=`dirname $0`
cd $sd/../

SRC=`pwd`
OUT=$1

## helpers

# download a file if it does not exist or if it is outdated
function gg_wget {
    local out=$1
    local url=$2

    local cwd=`pwd`

    mkdir -p $out
    cd $out

    # should not re-download if file is the same
    wget -N $url

    cd $cwd
}

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

# ctest_debug

function gg_run_ctest_debug {
    cd $SRC

    rm -rf build-ci && mkdir build-ci && cd build-ci

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Debug ..     ) 2>&1 | tee $OUT/${ci}-cmake.log
    (time make -j4                              ) 2>&1 | tee $OUT/${ci}-make.log
    (time ctest --output-on-failure -E test-opt ) 2>&1 | tee $OUT/${ci}-ctest.log

    set +e
}

function gg_sum_ctest_debug {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest in debug mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
    gg_printf '\n'
}

# ctest_release

function gg_run_ctest_release {
    cd $SRC

    rm -rf build-ci && mkdir build-ci && cd build-ci

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Release ..   ) 2>&1 | tee $OUT/${ci}-cmake.log
    (time make -j4                              ) 2>&1 | tee $OUT/${ci}-make.log
    (time ctest --output-on-failure -E test-opt ) 2>&1 | tee $OUT/${ci}-ctest.log

    set +e
}

function gg_sum_ctest_release {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest in release mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
}

# gpt_2

function gg_run_gpt_2 {
    cd $SRC

    gg_wget models/gpt-2 https://huggingface.co/ggerganov/ggml/resolve/main/ggml-model-gpt-2-117M.bin

    rm -rf build-ci && mkdir build-ci && cd build-ci

    set -e

    cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4

    ./bin/gpt-2 --model ../models/gpt-2/ggml-model-gpt-2-117M.bin -s 1234 -n 64 -t 4 > $OUT/${ci}.log 2>&1

    set +e
}

function gg_sum_gpt_2 {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs short GPT-2 text generation\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}.log)"
    gg_printf '```\n'
}

## main

ret=0

$ret && gg_run ctest_debug
$ret && gg_run ctest_release
$ret && gg_run gpt_2

exit $ret
