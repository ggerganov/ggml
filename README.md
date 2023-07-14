# ggml

[Roadmap](https://github.com/users/ggerganov/projects/7) / [Manifesto](https://github.com/ggerganov/llama.cpp/discussions/205)

Tensor library for machine learning

***Note that this project is under active development. \
Some of the development is currently happening in the [llama.cpp](https://github.com/ggerganov/llama.cpp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) repos***

## Features

- Written in C
- 16-bit float support
- Integer quantization support (4-bit, 5-bit, 8-bit, etc.)
- Automatic differentiation
- ADAM and L-BFGS optimizers
- Optimized for Apple Silicon
- On x86 architectures utilizes AVX / AVX2 intrinsics
- No third-party dependencies
- Zero memory allocations during runtime

## Updates

- [X] Example of GPT-2 inference [examples/gpt-2](https://github.com/ggerganov/ggml/tree/master/examples/gpt-2)
- [X] Example of GPT-J inference [examples/gpt-j](https://github.com/ggerganov/ggml/tree/master/examples/gpt-j)
- [X] Example of Whisper inference [examples/whisper](https://github.com/ggerganov/ggml/tree/master/examples/whisper)
- [X] Support 4-bit integer quantization https://github.com/ggerganov/ggml/pull/27
- [X] Example of Cerebras-GPT inference [examples/gpt-2](https://github.com/ggerganov/ggml/tree/master/examples/gpt-2)
- [ ] Example of FLAN-T5 inference https://github.com/ggerganov/ggml/pull/12
- [X] Example of LLaMA inference [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- [X] Example of LLaMA training [ggerganov/llama.cpp/examples/baby-llama](https://github.com/ggerganov/llama.cpp/tree/master/examples/baby-llama)
- [X] Example of Falcon inference [cmp-nct/ggllm.cpp](https://github.com/cmp-nct/ggllm.cpp)
- [X] Example of BLOOM inference [NouamaneTazi/bloomz.cpp](https://github.com/NouamaneTazi/bloomz.cpp)
- [X] Example of RWKV inference [saharNooby/rwkv.cpp](https://github.com/saharNooby/rwkv.cpp)
- [ ] Example of [SAM](https://github.com/facebookresearch/segment-anything) inference
- [X] Idea for GPU support: https://github.com/ggerganov/llama.cpp/discussions/915
- [X] Example of StableLM (GPT-NeoX) inference [examples/gpt-neox](https://github.com/ggerganov/ggml/tree/master/examples/gpt-neox)
- [X] Example of BERT inference [skeskinen/bert.cpp](https://github.com/skeskinen/bert.cpp)
- [X] Example of 💫 StarCoder inference [examples/starcoder](https://github.com/ggerganov/ggml/tree/master/examples/starcoder)
- [X] Example of MPT inference [examples/mpt](https://github.com/ggerganov/ggml/tree/master/examples/mpt)
- [X] Example of Replit inference [examples/replit](https://github.com/ggerganov/ggml/tree/master/examples/replit)
- [X] Example of BioGPT inference [PABannier/biogpt.cpp](https://github.com/PABannier/biogpt.cpp)
- [X] Example of Encodec inference [PABannier/encodec.cpp](https://github.com/PABannier/encodec.cpp) 
- [X] Example of CLIP inference [monatis/clip.cpp](https://github.com/monatis/clip.cpp)

## Whisper inference (example)

With ggml you can efficiently run [Whisper](examples/whisper) inference on the CPU.

Memory requirements:

| Model  | Disk   | Mem     |
| ---    | ---    | ---     |
| tiny   |  75 MB | ~280 MB |
| base   | 142 MB | ~430 MB |
| small  | 466 MB | ~1.0 GB |
| medium | 1.5 GB | ~2.6 GB |
| large  | 2.9 GB | ~4.7 GB |

## GPT inference (example)

With ggml you can efficiently run [GPT-2](examples/gpt-2) and [GPT-J](examples/gpt-j) inference on the CPU.

Here is how to run the example programs:

```bash
# Build ggml + examples
git clone https://github.com/ggerganov/ggml
cd ggml
mkdir build && cd build
cmake ..
make -j4 gpt-2 gpt-j

# Run the GPT-2 small 117M model
../examples/gpt-2/download-ggml-model.sh 117M
./bin/gpt-2 -m models/gpt-2-117M/ggml-model.bin -p "This is an example"

# Run the GPT-J 6B model (requires 12GB disk space and 16GB CPU RAM)
../examples/gpt-j/download-ggml-model.sh 6B
./bin/gpt-j -m models/gpt-j-6B/ggml-model.bin -p "This is an example"

# Install Python dependencies
python3 -m pip install -r ../requirements.txt

# Run the Cerebras-GPT 111M model
# Download from: https://huggingface.co/cerebras
python3 ../examples/gpt-2/convert-cerebras-to-ggml.py /path/to/Cerebras-GPT-111M/
./bin/gpt-2 -m /path/to/Cerebras-GPT-111M/ggml-model-f16.bin -p "This is an example"
```

The inference speeds that I get for the different models on my 32GB MacBook M1 Pro are as follows:

| Model | Size  | Time / Token |
| ---   | ---   | ---    |
| GPT-2 |  117M |   5 ms |
| GPT-2 |  345M |  12 ms |
| GPT-2 |  774M |  23 ms |
| GPT-2 | 1558M |  42 ms |
| ---   | ---   | ---    |
| GPT-J |    6B | 125 ms |

For more information, checkout the corresponding programs in the [examples](examples) folder.

## Using cuBLAS

```bash
# fix the path to point to your CUDA compiler
cmake -DGGML_CUBLAS=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc ..
```

## Using clBLAST

```bash
cmake -DGGML_CLBLAST=ON ..
```

## Resources

- [GGML - Large Language Models for Everyone](https://github.com/rustformers/llm/blob/main/crates/ggml/README.md): a description of the GGML format provided by the maintainers of the `llm` Rust crate, which provides Rust bindings for GGML
- [marella/ctransformers](https://github.com/marella/ctransformers): Python bindings for GGML models.
- [go-skynet/go-ggml-transformers.cpp](https://github.com/go-skynet/go-ggml-transformers.cpp): Golang bindings for GGML models
- [smspillaz/ggml-gobject](https://github.com/smspillaz/ggml-gobject): GObject-introspectable wrapper for use of GGML on the GNOME platform.
