#!/bin/bash

# This script downloads GPT-J model files that have already been converted to ggml format.
# This way you don't have to convert them yourself.
#
# If you want to download the original GPT-J model files, use the "download-model.sh" script instead.

#src="https://ggml.ggerganov.com"
#pfx="ggml-model-gpt-j"

src="https://huggingface.co/ggerganov/ggml"
pfx="resolve/main/ggml-model-gpt-j"

ggml_path=$(dirname $(realpath $0))

# GPT-J models
models=( "6B" )

# list available models
function list_models {
    printf "\n"
    printf "  Available models:"
    for model in "${models[@]}"; do
        printf " $model"
    done
    printf "\n\n"
}

if [ "$#" -ne 1 ]; then
    printf "Usage: $0 <model>\n"
    list_models

    exit 1
fi

model=$1

if [[ ! " ${models[@]} " =~ " ${model} " ]]; then
    printf "Invalid model: $model\n"
    list_models

    exit 1
fi

# download ggml model

printf "Downloading ggml model $model ...\n"

mkdir -p models/gpt-j-$model

if [ -x "$(command -v wget)" ]; then
    wget --quiet --show-progress -O models/gpt-j-$model/ggml-model.bin $src/$pfx-$model.bin
elif [ -x "$(command -v curl)" ]; then
    curl -L --output models/gpt-j-$model/ggml-model.bin $src/$pfx-$model.bin
else
    printf "Either wget or curl is required to download models.\n"
    exit 1
fi

if [ $? -ne 0 ]; then
    printf "Failed to download ggml model $model \n"
    printf "Please try again later or download the original GPT-J model files and convert them yourself.\n"
    exit 1
fi

printf "Done! Model '$model' saved in 'models/gpt-j-$model/ggml-model.bin'\n"
printf "You can now use it like this:\n\n"
printf "  $ ./bin/gpt-j -m models/gpt-j-$model/ggml-model.bin -p \"This is an example\"\n"
printf "\n"
