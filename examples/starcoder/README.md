# 💫 StarCoder

This is a C++ example running 💫 StarCoder inference using the [ggml](https://github.com/ggerganov/ggml) library.

The program runs on the CPU - no video card is required.

The example supports the following 💫 StarCoder models:

- `bigcode/starcoder`
- `bigcode/gpt_bigcode-santacoder` aka the smol StarCoder

Sample performance on MacBook M1 Pro:

TODO


Sample output:

```
$ ./bin/starcoder -h
usage: ./bin/starcoder [options]

options:
  -h, --help            show this help message and exit
  -s SEED, --seed SEED  RNG seed (default: -1)
  -t N, --threads N     number of threads to use during computation (default: 8)
  -p PROMPT, --prompt PROMPT
                        prompt to start generation with (default: random)
  -n N, --n_predict N   number of tokens to predict (default: 200)
  --top_k N             top-k sampling (default: 40)
  --top_p N             top-p sampling (default: 0.9)
  --temp N              temperature (default: 1.0)
  -b N, --batch_size N  batch size for prompt processing (default: 8)
  -m FNAME, --model FNAME
                        model path (default: models/starcoder-117M/ggml-model.bin)

$ ./bin/starcoder -m ../models/bigcode/gpt_bigcode-santacoder-ggml-q4_1.bin -p "def fibonnaci(" -t 4 --top_k 0 --top_p 0.95 --temp 0.2      
main: seed = 1683881276
starcoder_model_load: loading model from '../models/bigcode/gpt_bigcode-santacoder-ggml-q4_1.bin'
starcoder_model_load: n_vocab = 49280
starcoder_model_load: n_ctx   = 2048
starcoder_model_load: n_embd  = 2048
starcoder_model_load: n_head  = 16
starcoder_model_load: n_layer = 24
starcoder_model_load: ftype   = 3
starcoder_model_load: ggml ctx size = 1794.90 MB
starcoder_model_load: memory size =   768.00 MB, n_mem = 49152
starcoder_model_load: model size  =  1026.83 MB
main: prompt: 'def fibonnaci('
main: number of tokens in prompt = 7, first 8 tokens: 563 24240 78 2658 64 2819 7 

def fibonnaci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(fibo(10))

main: mem per token =  9597928 bytes
main:     load time =   480.43 ms
main:   sample time =    26.21 ms
main:  predict time =  3987.95 ms / 19.36 ms per token
main:    total time =  4580.56 ms
```

## Quick start
```bash
git clone https://github.com/ggerganov/ggml
cd ggml

# Install Python dependencies
python3 -m pip install -r requirements.txt

# Convert HF model to ggml
python examples/starcoder/convert-hf-to-ggml.py bigcode/gpt_bigcode-santacoder

# Build ggml + examples
mkdir build && cd build
cmake .. && make -j4 starcoder starcoder-quantize

# quantize the model
./bin/starcoder-quantize ../models/bigcode/gpt_bigcode-santacoder-ggml.bin ../models/bigcode/gpt_bigcode-santacoder-ggml-q4_1.bin 3

# run inference
./bin/starcoder -m ../models/bigcode/gpt_bigcode-santacoder-ggml-q4_1.bin -p "def fibonnaci(" --top_k 0 --top_p 0.95 --temp 0.2
```


## Downloading and converting the original models (💫 StarCoder)

You can download the original model and convert it to `ggml` format using the script `convert-hf-to-ggml.py`:

```
# Convert HF model to ggml
python examples/starcoder/convert-hf-to-ggml.py bigcode/gpt_bigcode-santacoder
```

This conversion requires that you have python and Transformers installed on your computer.

## Quantizing the models

You can also try to quantize the `ggml` models via 4-bit integer quantization.

```
# quantize the model
./bin/starcoder-quantize ../models/bigcode/gpt_bigcode-santacoder-ggml.bin ../models/bigcode/gpt_bigcode-santacoder-ggml-q4_1.bin 3
```

| Model | Original size | Quantized size | Quantization type |
| --- | --- | --- | --- |
| `bigcode/gpt_bigcode-santacoder` | 5396.45 MB | 1026.83 MB | 4-bit integer (q4_1) |
| `bigcode/starcoder` | 71628.23 MB | 13596.23 MB | 4-bit integer (q4_1) |
