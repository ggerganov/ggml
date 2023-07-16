#/bin/bash

sd=`dirname $0`
cd $sd/../

SRC=`pwd`
OUT=$1

## ci

function gg_ci_0 {
    cd $SRC

    mkdir build-ci-0
    cd build-ci-0

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Debug .. ) 2>&1 | tee $OUT/ci-0-cmake.log
    (time make -j4                          ) 2>&1 | tee $OUT/ci-0-make.log
    (time ctest -E test-opt                 ) 2>&1 | tee $OUT/ci-0-ctest.log

    set +e
}

function gg_ci_1 {
    cd $SRC

    mkdir build-ci-1
    cd build-ci-1

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Debug .. ) 2>&1 | tee $OUT/ci-1-cmake.log
    (time make -j4                          ) 2>&1 | tee $OUT/ci-1-make.log
    (time ctest -E test-opt                 ) 2>&1 | tee $OUT/ci-1-ctest.log

    set +e
}

## main

ret=0

set -o pipefail

gg_ci_0 | tee $OUT/ci-0.log
cur=$?
echo "$cur" > $OUT/ci-0.exit
ret=$(($ret + $cur))

gg_ci_1 | tee $OUT/ci-1.log
cur=$?
echo "$cur" > $OUT/ci-1.exit
ret=$(($ret + $cur))

set +o pipefail

## generate README.md with summary

status="$(if [ $ret -eq 0 ]; then echo "SUCCESS"; else echo "FAILURE ($ret)"; fi)"

gg_printf README.md "## Summary\n\n"

gg_printf README.md "- status: ${status}\n"
gg_printf README.md "- date:   $(date)\n"
gg_printf README.md "- repo:   ${GG_CI_REPO}\n"
gg_printf README.md "- author: ${GG_CI_COMMIT_AUTHOR}\n"
gg_printf README.md "- commit: ${GG_CI_COMMIT_URL}\n"
gg_printf README.md "\`\`\`\n${GG_CI_COMMIT_MSG}\n\`\`\`\n"
gg_printf README.md "\n"

gg_printf README.md "### ci-0\n\n"

gg_printf README.md "\`\`\`\n"
gg_printf README.md "$(cat $OUT/ci-0-ctest.log)"
gg_printf README.md "\`\`\`\n"
gg_printf README.md "\n"

gg_printf README.md "### ci-1\n\n"

gg_printf README.md "\`\`\`\n"
gg_printf README.md "$(cat $OUT/ci-1-ctest.log)"
gg_printf README.md "\`\`\`\n"

exit $ret
