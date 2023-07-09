# MPT

Ref: https://github.com/mosaicml/llm-foundry#mpt

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
git clone https://huggingface.co/mosaicml/mpt-30b

# convert model to FP16
python3 ../examples/mpt/convert-h5-to-ggml.py ./mpt-30b 1

# run inference using FP16 precision
./bin/mpt -m ./mpt-30b/ggml-model-f16.bin -p "I believe the meaning of life is" -t 8 -n 64

# quantize the model to 5-bits using Q5_0 quantization
./bin/mpt-quantize ./mpt-30b/ggml-model-f16.bin ./mpt-30b/ggml-model-q5_0.bin q5_0
```
