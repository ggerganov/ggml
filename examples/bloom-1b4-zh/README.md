# Bloom-1b4-zh

Ref: https://huggingface.co/Langboat/bloom-1b4-zh

## Usage

```bash
# get the repo and build it
git clone https://github.com/ggerganov/ggml
cd ggml
mkdir build && cd build
cmake ..
make -j

# get the model from HuggingFace
# be sure to have git-lfs installed
git clone https://huggingface.co/Langboat/bloom-1b4-zh

# convert model to FP16
mkdir -p output
python3 ../examples/bloom-1b4-zh/convert-h5-to-ggml.py ./bloom-1b4-zh ./output

# run inference using FP16 precision
./bin/bloom-1b4-zh -m ./output/ggml-model-f16.bin -p "I believe the meaning of life is" -t 8 -n 64

# quantize the model to 5-bits using Q5_0 quantization
./bin/bloom-1b4-zh-quantize ./output/ggml-model-f16.bin ./output/ggml-model-q5_0.bin q5_0
