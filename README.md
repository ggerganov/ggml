# ggml

Tensor library in C for machine learning

## Features

- Automatic differentiation (WIP)
- 16-bit float support
- ADAM and L-BFGS optimizers
- Optimized for Arm64 architectures (i.e. MacBook M1) via NEON intrinsics
- On x86 architectures utilzes AVX intrinsics
- No third-party dependencies
- Zero memory allocations during runtime

## Local GPT inference

Using ggml you can run [GPT-2](examples/gpt-2) and [GPT-J](examples/gpt-j) inference locally on your computer without any additional software or hardware. You don't even need to install python or any other third-party library.

The example programs are implemented in C++. They run entirely on the CPU.

Here is how to use them:

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
```

This is the inference speed for the different models on my MacBook M1 Pro:

| Model | Size  | Time / Token |
| ---   | ---   | ---    |
| GPT-2 |  117M |   5 ms |
| GPT-2 |  345M |  12 ms |
| GPT-2 |  774M |  23 ms |
| GPT-2 | 1558M |  42 ms |
| ---   | ---   | ---    |
| GPT-J |    6B | 125 ms |

For more information, checkout the corresponding programs in the [examples](examples) folder.
