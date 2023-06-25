# GPT-NeoX

Transformer architecture: GPT-NeoX

Ref: https://github.com/stability-AI/stableLM/#stablelm-alpha

## Usage

```bash
# get the repo and build it
git clone https://github.com/ggerganov/ggml
cd ggml
mkdir build && cd build
cmake ..
make -j

# get the StableLM 3B Alpha model
git clone https://huggingface.co/stabilityai/gpt_neox-base-alpha-3b

# install Python dependencies
python3 -m pip install -r ../requirements.txt

# convert model to FP16
python3 ../examples/gpt_neox/convert-h5-to-ggml.py ./stablelm-base-alpha-3b/ 1

# run inference using FP16 precision
make -j && ./bin/gpt_neox -m ./stablelm-base-alpha-3b/ggml-model-f16.bin -p "I believe the meaning of life is" -t 8 -n 64

main: seed = 1681940611
gpt_neox_model_load: loading model from 'models/stablelm-base-alpha-3b/ggml-model-f16.bin' - please wait ...
gpt_neox_model_load: n_vocab = 50688
gpt_neox_model_load: n_ctx   = 4096
gpt_neox_model_load: n_embd  = 4096
gpt_neox_model_load: n_head  = 32
gpt_neox_model_load: n_layer = 16
gpt_neox_model_load: n_rot   = 32
gpt_neox_model_load: ftype   = 1
gpt_neox_model_load: ggml ctx size = 10011.10 MB
gpt_neox_model_load: memory_size =  2048.00 MB, n_mem = 65536
gpt_neox_model_load: ................................ done
gpt_neox_model_load: model size =  6939.28 MB / num tensors = 260
main: number of tokens in prompt = 7
main: token[0] =     42, I
main: token[1] =   2868,  believe
main: token[2] =    253,  the
main: token[3] =   4495,  meaning
main: token[4] =    273,  of
main: token[5] =   1495,  life
main: token[6] =    310,  is

I believe the meaning of life is to grow, to find a way, to love, to find an appreciation for life, and to live it with all of its beauty.

For I am the child of God. I am the offspring of God's love. I am the offspring of the light of the world. I am the offspring of the

main: mem per token = 12186760 bytes
main:     load time =  2118.55 ms
main:   sample time =     9.59 ms
main:  predict time =  4474.07 ms / 63.92 ms per token
main:    total time =  6911.26 ms
```

## 5-bit integer quantization mode

```bash
# quantize the model to 5-bits using Q5_0 quantization
./bin/gpt_neox-quantize ./stablelm-base-alpha-3b/ggml-model-f16.bin ./stablelm-base-alpha-3b/ggml-model-q5_0.bin q5_0

# run the quantized model
./bin/gpt_neox -m ./stablelm-base-alpha-3b/ggml-model-q5_0.bin -p "I believe the meaning of life is" -t 8 -n 64

main: seed = 1682021489
gpt_neox_model_load: loading model from 'models/stablelm-base-alpha-3b/ggml-model-q5_0.bin' - please wait ...
gpt_neox_model_load: n_vocab = 50688
gpt_neox_model_load: n_ctx   = 4096
gpt_neox_model_load: n_embd  = 4096
gpt_neox_model_load: n_head  = 32
gpt_neox_model_load: n_layer = 16
gpt_neox_model_load: n_rot   = 32
gpt_neox_model_load: ftype   = 6
gpt_neox_model_load: ggml ctx size = 5676.10 MB
gpt_neox_model_load: memory_size =  1024.00 MB, n_mem = 65536
gpt_neox_model_load: ........................ done
gpt_neox_model_load: model size =  2604.28 MB / num tensors = 196
main: number of tokens in prompt = 7
main: token[0] =     42, I
main: token[1] =   2868,  believe
main: token[2] =    253,  the
main: token[3] =   4495,  meaning
main: token[4] =    273,  of
main: token[5] =   1495,  life
main: token[6] =    310,  is

I believe the meaning of life is to love and be loved. The last three verses were enough to tie us all together. If you love someone you love them all. There are some things in this world that are just not equal in Heaven. - Be here in this moment.

This world is not what is outside of us. It is what

main: mem per token = 12958024 bytes
main:     load time =   850.51 ms
main:   sample time =     9.95 ms
main:  predict time =  3103.81 ms / 44.34 ms per token
main:    total time =  4177.68 ms

```

## Notes

- No guarantees for correctness
- The tokenizer is currently hacked - probably works only for English
- Non-parallel residual is not supported
- Contributions and improvements are welcome
