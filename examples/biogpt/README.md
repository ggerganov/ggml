# biogpt.cpp

Inference of BioGPT model in pure C/C++.

## Description

The main goal of `biogpt.cpp` is to run the [BioGPT](https://arxiv.org/abs/2210.10341) model using 4-bit quantization on a MacBook.

---

Here is a typical run using BioGPT:

```java
./biogpt -p "trastuzumab"
main: seed = 1684061910
biogpt_model_load: loading model from './ggml_weights/ggml-model.bin'
biogpt_model_load: n_vocab       = 42384
biogpt_model_load: d_ff          = 4096
biogpt_model_load: d_model       = 1024
biogpt_model_load: n_positions   = 1024
biogpt_model_load: n_head        = 16
biogpt_model_load: n_layer       = 24
biogpt_model_load: f16           = 0
biogpt_model_load: ggml ctx size = 1888.36 MB
biogpt_model_load: memory size =   192.00 MB, n_mem = 24576
biogpt_model_load: model size    = 1488.36 MB
main: prompt: 'Trastuzumab'
main: number of tokens in prompt = 4, first 8 tokens: 2 7548 1171 32924

Trastuzumab (Herceptin) is the first-line treatment for HER2-positive breast cancer and is the only agent approved by the
US Food and Drug Administration for the treatment of HER2-positive metastatic breast cancer. In the US, approximately 20 %
of patients with HER2-positive metastatic breast cancer fail to achieve response to first-line treatment with trastuzumab.
This article discusses the mechanisms of trastuzumab resistance , strategies for overcoming trastuzumab resistance, and the
potential role of other targeted therapies. New treatment options for multiple myeloma. The past 2 years have seen 
significant advances in the treatment of multiple myeloma, particularly with the introduction of novel agents, particularly
the proteasome inhibitors and immunomodulatory drugs. These new agents are more effective and are associated with fewer
side effects than the older drugs. Their use has improved survival, with recent clinical trials evaluating combination
therapies with novel agents. However, their role in the treatment of multiple myeloma remains unclear and remains to be
evaluated in future clinical trials.

main: mem per token =  4911704 bytes
main:     load time =   456.57 ms
main:   sample time =    23.32 ms
main:  predict time =  4140.06 ms / 20.39 ms per token
main:    total time =  4672.20 ms
```

## Memory requirements and speed

The inference speeds that I get for the different quantized models on my 16GB MacBook M1 Pro are as follows:

| Model | Size  | Time / Token |
| ---   | ---   | ---  |
| Original |  1.5G |   20 ms |
| Q4_0 | 240M |  8 ms |
| Q4_1 | 286M |  9 ms |
| Q5_0 | 265M |  10 ms |
| Q5_1 | 288M |  11 ms |
| Q8_0 | 432M |  10 ms |


## API

```java
./biogpt -h
usage: ./biogpt [options]

options:
  -h, --help            show this help message and exit
  -s SEED, --seed SEED  RNG seed (default: -1)
  -t N, --threads N     number of threads to use during computation (default: 4)
  -p PROMPT, --prompt PROMPT
                        prompt to start generation with (default: random)
  -l LANG               language of the prompt          (default: )
  -n N, --n_predict N   number of tokens to predict (default: 200)
  --top_k N             top-k sampling (default: 40)
  --top_p N             top-p sampling (default: 0.9)
  --temp N              temperature (default: 0.9)
  -b N, --batch_size N  batch size for prompt processing (default: 8)
  -m FNAME, --model FNAME
                        model path (default: ./ggml_weights/ggml-model.bin)
```