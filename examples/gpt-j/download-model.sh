#!/bin/bash

printf "To obtain the GPT-J 6B model files, please visit: https://huggingface.co/EleutherAI/gpt-j-6B\n\n"

printf "The model is very big. For example, the reposirory above is 72GB in size.\n"
printf "If you are sure that you want to clone it, simply run the following command:\n\n"

printf " $ git clone https://huggingface.co/EleutherAI/gpt-j-6B models/gpt-j-6B\n\n"

printf "Alternatively, use the 'download-ggml-model.sh' script to download a 12GB ggml version of the model.\n"
printf "This version is enough to run inference using the ggml library.\n\n"
