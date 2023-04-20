# StableLM

Transformer architecture: GPTNeoX

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
git clone https://huggingface.co/stabilityai/stablelm-base-alpha-3b

# convert model to FP16
python3 ../examples/stablelm/convert-h5-to-ggml.py ./stablelm-base-alpha-3b/ 1

# run inference using FP16 precision
make -j && ./bin/stablelm -m ./stablelm-base-alpha-3b/ggml-model-f16.bin -p "I believe the meaning of life is" -t 8 -n 64

main: seed = 1681940611
stablelm_model_load: loading model from 'models/stablelm-base-alpha-3b/ggml-model-f16.bin' - please wait ...
stablelm_model_load: n_vocab = 50688
stablelm_model_load: n_ctx   = 4096
stablelm_model_load: n_embd  = 4096
stablelm_model_load: n_head  = 32
stablelm_model_load: n_layer = 16
stablelm_model_load: n_rot   = 32
stablelm_model_load: ftype   = 1
stablelm_model_load: ggml ctx size = 10011.10 MB
stablelm_model_load: memory_size =  2048.00 MB, n_mem = 65536
stablelm_model_load: ................................ done
stablelm_model_load: model size =  6939.28 MB / num tensors = 260
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

## Run in 4-bit integer quantization mode

```
# quantize the model to 4-bits using Q4_3 quantization
./bin/stablelm-quantize ./stablelm-base-alpha-3b/ggml-model-f16.bin ./stablelm-base-alpha-3b/ggml-model-q4_3.bin 6

# run the quantized model
./bin/stablelm -m ./stablelm-base-alpha-3b/ggml-model-q4_3.bin -p "I believe the meaning of life is" -t 8 -n 64

main: seed = 1682021489
stablelm_model_load: loading model from 'models/stablelm-base-alpha-3b/ggml-model-q4_3.bin' - please wait ...
stablelm_model_load: n_vocab = 50688
stablelm_model_load: n_ctx   = 4096
stablelm_model_load: n_embd  = 4096
stablelm_model_load: n_head  = 32
stablelm_model_load: n_layer = 16
stablelm_model_load: n_rot   = 32
stablelm_model_load: ftype   = 6
stablelm_model_load: ggml ctx size = 5676.10 MB
stablelm_model_load: memory_size =  1024.00 MB, n_mem = 65536
stablelm_model_load: ........................ done
stablelm_model_load: model size =  2604.28 MB / num tensors = 196
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
