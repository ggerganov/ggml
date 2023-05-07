# RedPajama

Transformer architecture: GPT-NeoX

Ref: https://github.com/togethercomputer/RedPajama-Data

## Usage

```bash
# get the repo and build it
git clone https://github.com/ggerganov/ggml
cd ggml
mkdir build && cd build
cmake ..
make -j

# get the RedPajama 3B Base model
git clone https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1

# convert model to FP16
python3 ../examples/redpajama/convert-h5-to-ggml.py ./RedPajama-INCITE-Base-3B-v1/ 1

# run inference using FP16 precision
make -j && ./bin/redpajama -m ./RedPajama-INCITE-Base-3B-v1/ggml-model-f16.bin -p "I believe the meaning of life is" -t 8 -n 64

main: seed = 1683382277
redpajama_model_load: loading model from './RedPajama-INCITE-Base-3B-v1/ggml-model-f16.bin' - please wait ...
redpajama_model_load: n_vocab = 50432
redpajama_model_load: n_ctx   = 2048
redpajama_model_load: n_embd  = 2560
redpajama_model_load: n_head  = 32
redpajama_model_load: n_layer = 32
redpajama_model_load: n_rot   = 80
redpajama_model_load: ftype   = 1
redpajama_model_load: ggml ctx size = 7376.40 MB
redpajama_model_load: memory_size =   640.00 MB, n_mem = 65536
redpajama_model_load: ................................................ done
redpajama_model_load: model size =  5296.58 MB / num tensors = 388
main: number of tokens in prompt = 7
main: token[0] =     42, I
main: token[1] =   2868,  believe
main: token[2] =    253,  the
main: token[3] =   4495,  meaning
main: token[4] =    273,  of
main: token[5] =   1495,  life
main: token[6] =    310,  is

I believe the meaning of life is to find your gift. A gift that is given to you by God.” – Audrey Hepburn
“Life’s most persistent and urgent question is, ‘What are you doing for others?’” – Martin Luther King, Jr.
“You have to figure out what you’re supposed to be doing.”

main: mem per token = 16137360 bytes
main:     load time =  4786.43 ms
main:   sample time =    15.11 ms
main:  predict time = 17114.62 ms / 244.49 ms per token
main:    total time = 22845.03 ms
```

## 4-bit integer quantization mode

```bash
# quantize the model to 4-bits using Q4_3 quantization
./bin/redpajama-quantize RedPajama-INCITE-Base-3B-v1/ggml-model-f16.bin RedPajama-INCITE-Base-3B-v1/ggml-model-q4_0.bin 2

# run the quantized model
./bin/redpajama -m ./RedPajama-INCITE-Base-3B-v1/ggml-model-q4_0.bin -p "I believe the meaning of life is" -t 8 -n 64

main: seed = 1683383230
redpajama_model_load: loading model from './RedPajama-INCITE-Base-3B-v1/ggml-model-q4_0.bin' - please wait ...
redpajama_model_load: n_vocab = 50432
redpajama_model_load: n_ctx   = 2048
redpajama_model_load: n_embd  = 2560
redpajama_model_load: n_head  = 32
redpajama_model_load: n_layer = 32
redpajama_model_load: n_rot   = 80
redpajama_model_load: ftype   = 2
redpajama_model_load: ggml ctx size = 3737.80 MB
redpajama_model_load: memory_size =   640.00 MB, n_mem = 65536
redpajama_model_load: ................................................ done
redpajama_model_load: model size =  1657.99 MB / num tensors = 388
main: number of tokens in prompt = 7
main: token[0] =     42, I
main: token[1] =   2868,  believe
main: token[2] =    253,  the
main: token[3] =   4495,  meaning
main: token[4] =    273,  of
main: token[5] =   1495,  life
main: token[6] =    310,  is

I believe the meaning of life is to love, to love deeply, to be able to love, to love without conditions, to love without measure, to love unconditionally, and to love until death.
The story of the song, I think, is that my love was never returned by my husband. He was a good man, but a

main: mem per token = 16128400 bytes
main:     load time =   841.67 ms
main:   sample time =    13.05 ms
main:  predict time =  9297.83 ms / 132.83 ms per token
main:    total time = 10695.10 ms
```

## Notes

- No guarantees for correctness
- The tokenizer is currently hacked - probably works only for English
- Contributions and improvements are welcome
