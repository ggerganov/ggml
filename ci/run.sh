#/bin/bash
#
# sample usage:
#
# mkdir tmp
#
# # CPU-only build
# bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with CUDA support
# GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#

if [ -z "$2" ]; then
    echo "usage: $0 <output-dir> <mnt-dir>"
    exit 1
fi

mkdir -p "$1"
mkdir -p "$2"

OUT=$(realpath "$1")
MNT=$(realpath "$2")

rm -v $OUT/*.log
rm -v $OUT/*.exit
rm -v $OUT/*.md

sd=`dirname $0`
cd $sd/../
SRC=`pwd`

## helpers

# download a file if it does not exist or if it is outdated
function gg_wget {
    local out=$1
    local url=$2

    local cwd=`pwd`

    mkdir -p $out
    cd $out

    # should not re-download if file is the same
    wget -nv -N $url

    cd $cwd
}

function gg_printf {
    printf -- "$@" >> $OUT/README.md
}

function gg_run {
    ci=$1

    set -o pipefail
    set -x

    gg_run_$ci | tee $OUT/$ci.log
    cur=$?
    echo "$cur" > $OUT/$ci.exit

    set +x
    set +o pipefail

    gg_sum_$ci

    ret=$((ret | cur))
}

## ci

# ctest_debug

function gg_run_ctest_debug {
    cd ${SRC}

    rm -rf build-ci-debug && mkdir build-ci-debug && cd build-ci-debug

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Debug ..     ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j                               ) 2>&1 | tee -a $OUT/${ci}-make.log

    (time ctest --output-on-failure -E test-opt ) 2>&1 | tee -a $OUT/${ci}-ctest.log

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
    cd ${SRC}

    rm -rf build-ci-release && mkdir build-ci-release && cd build-ci-release

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Release ..   ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j                               ) 2>&1 | tee -a $OUT/${ci}-make.log

    if [ -z $GG_BUILD_LOW_PERF ]; then
        (time ctest --output-on-failure ) 2>&1 | tee -a $OUT/${ci}-ctest.log
    else
        (time ctest --output-on-failure -E test-opt ) 2>&1 | tee -a $OUT/${ci}-ctest.log
    fi

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
    cd ${SRC}

    gg_wget models-mnt/gpt-2 https://huggingface.co/ggerganov/ggml/resolve/main/ggml-model-gpt-2-117M.bin

    cd build-ci-release

    set -e

    model="../models-mnt/gpt-2/ggml-model-gpt-2-117M.bin"
    prompts="../examples/prompts/gpt-2.txt"

    (time ./bin/gpt-2 --model ${model} -s 1234 -n 64 -tt ${prompts}                       ) 2>&1 | tee -a $OUT/${ci}-tg.log
    (time ./bin/gpt-2 --model ${model} -s 1234 -n 64 -p "I believe the meaning of life is") 2>&1 | tee -a $OUT/${ci}-tg.log

    set +e
}

function gg_sum_gpt_2 {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs short GPT-2 text generation\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-tg.log)"
    gg_printf '```\n'
}

# mnist

function gg_run_mnist {
    cd ${SRC}

    cd build-ci-release

    set -e

    mkdir -p models/mnist
    python3 ../examples/mnist/convert-h5-to-ggml.py ../examples/mnist/models/mnist/mnist_model.state_dict

    model_f32="./models/mnist/ggml-model-f32.bin"
    samples="../examples/mnist/models/mnist/t10k-images.idx3-ubyte"

    # first command runs and exports "mnist.ggml", the second command runs the exported model

    (time ./bin/mnist     ${model_f32} ${samples} ) 2>&1 | tee -a $OUT/${ci}-mnist.log
    (time ./bin/mnist-cpu ./mnist.ggml ${samples} ) 2>&1 | tee -a $OUT/${ci}-mnist.log

    set +e
}

function gg_sum_mnist {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'MNIST\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-mnist.log)"
    gg_printf '```\n'
}

# whisper

function gg_run_whisper {
    cd ${SRC}

    gg_wget models-mnt/whisper/ https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
    gg_wget models-mnt/whisper/ https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav

    cd build-ci-release

    set -e

    path_models="../models-mnt/whisper/"
    model_f16="${path_models}/ggml-base.en.bin"
    audio_0="${path_models}/jfk.wav"

    (time ./bin/whisper -m ${model_f16} -f ${audio_0} 2>&1 | tee -a $OUT/${ci}-main.log) || true

    set +e
}

function gg_sum_whisper {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs short Whisper transcription\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-main.log)"
    gg_printf '```\n'
}

# mpt

function gg_run_mpt {
    cd ${SRC}

    gg_wget models-mnt/mpt/7B/ https://huggingface.co/mosaicml/mpt-7b/raw/main/config.json
    gg_wget models-mnt/mpt/7B/ https://huggingface.co/mosaicml/mpt-7b/raw/main/tokenizer.json
    gg_wget models-mnt/mpt/7B/ https://huggingface.co/mosaicml/mpt-7b/raw/main/tokenizer_config.json
    gg_wget models-mnt/mpt/7B/ https://huggingface.co/mosaicml/mpt-7b/raw/main/pytorch_model.bin.index.json
    gg_wget models-mnt/mpt/7B/ https://huggingface.co/mosaicml/mpt-7b/raw/main/configuration_mpt.py
    gg_wget models-mnt/mpt/7B/ https://huggingface.co/mosaicml/mpt-7b/resolve/main/pytorch_model-00001-of-00002.bin
    gg_wget models-mnt/mpt/7B/ https://huggingface.co/mosaicml/mpt-7b/resolve/main/pytorch_model-00002-of-00002.bin

    cd build-ci-release

    set -e

    path_models="../models-mnt/mpt/7B"
    model_f16="${path_models}/ggml-model-f16.bin"
    model_q4_0="${path_models}/ggml-model-q4_0.bin"

    python3 ../examples/mpt/convert-h5-to-ggml.py ${path_models} 1
    ./bin/mpt-quantize ${model_f16} ${model_q4_0} q4_0

    (time ./bin/mpt --model ${model_f16}  -s 1234 -n 64 -p "I believe the meaning of life is") 2>&1 | tee -a $OUT/${ci}-tg.log
    (time ./bin/mpt --model ${model_q4_0} -s 1234 -n 64 -p "I believe the meaning of life is") 2>&1 | tee -a $OUT/${ci}-tg.log

    set +e
}

function gg_sum_mpt {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs short MPT text generation\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-tg.log)"
    gg_printf '```\n'
}

## main

if [ -z $GG_BUILD_LOW_PERF ]; then
    rm -rf ${SRC}/models-mnt

    mnt_models=${MNT}/models
    mkdir -p ${mnt_models}
    ln -sfn ${mnt_models} ${SRC}/models-mnt
fi

python3 -m pip install -r ${SRC}/requirements.txt

ret=0

test $ret -eq 0 && gg_run ctest_debug
test $ret -eq 0 && gg_run ctest_release
test $ret -eq 0 && gg_run gpt_2
test $ret -eq 0 && gg_run mnist
test $ret -eq 0 && gg_run whisper

if [ -z $GG_BUILD_LOW_PERF ]; then
    test $ret -eq 0 && gg_run mpt
fi

exit $ret
