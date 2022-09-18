#!/bin/bash

ggml_path=$(dirname $(realpath $0))

# GPT-2 models
models=( "117M" "345M" "774M" "1558M" )

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

# download model

printf "Downloading model $model ...\n"

mkdir -p models/gpt-2-$model

for file in checkpoint encoder.json hparams.json model.ckpt.data-00000-of-00001 model.ckpt.index model.ckpt.meta vocab.bpe; do
    wget --quiet --show-progress -O models/gpt-2-$model/$file https://openaipublic.blob.core.windows.net/gpt-2/models/$model/$file
done

printf "Done! Model '$model' saved in 'models/gpt-2-$model/'\n\n"
printf "Run the convert-ckpt-to-ggml.py script to convert the model to ggml format.\n"
printf "\n"
printf "  python $ggml_path/convert-ckpt-to-ggml.py models/gpt-2-$model/\n"
printf "\n"
