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
- On ppc64 architectures utilizes VSX intrinsics
- No third-party dependencies
- Zero memory allocations during runtime

## Updates

- [X] Example of GPT-2 inference [examples/gpt-2](https://github.com/ggerganov/ggml/tree/master/examples/gpt-2)
- [X] Example of GPT-J inference [examples/gpt-j](https://github.com/ggerganov/ggml/tree/master/examples/gpt-j)
- [X] Example of Whisper inference [examples/whisper](https://github.com/ggerganov/ggml/tree/master/examples/whisper)
- [X] Example of LLaMA inference [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- [X] Example of LLaMA training [ggerganov/llama.cpp/examples/baby-llama](https://github.com/ggerganov/llama.cpp/tree/master/examples/baby-llama)
- [X] Example of Falcon inference [cmp-nct/ggllm.cpp](https://github.com/cmp-nct/ggllm.cpp)
- [X] Example of BLOOM inference [NouamaneTazi/bloomz.cpp](https://github.com/NouamaneTazi/bloomz.cpp)
- [X] Example of RWKV inference [saharNooby/rwkv.cpp](https://github.com/saharNooby/rwkv.cpp)
- [X] Example of SAM inference [examples/sam](https://github.com/ggerganov/ggml/tree/master/examples/sam)
- [X] Example of BERT inference [skeskinen/bert.cpp](https://github.com/skeskinen/bert.cpp)
- [X] Example of BioGPT inference [PABannier/biogpt.cpp](https://github.com/PABannier/biogpt.cpp)
- [X] Example of Encodec inference [PABannier/encodec.cpp](https://github.com/PABannier/encodec.cpp)
- [X] Example of CLIP inference [monatis/clip.cpp](https://github.com/monatis/clip.cpp)
- [X] Example of MiniGPT4 inference [Maknee/minigpt4.cpp](https://github.com/Maknee/minigpt4.cpp)
- [X] Example of ChatGLM inference [li-plus/chatglm.cpp](https://github.com/li-plus/chatglm.cpp)
- [X] Example of Stable Diffusion inference [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [X] Example of Qwen inference [QwenLM/qwen.cpp](https://github.com/QwenLM/qwen.cpp)
- [X] Example of YOLO inference [examples/yolo](https://github.com/ggerganov/ggml/tree/master/examples/yolo)
- [X] Example of ViT inference [staghado/vit.cpp](https://github.com/staghado/vit.cpp)
- [X] Example of multiple LLMs inference [foldl/chatllm.cpp](https://github.com/foldl/chatllm.cpp)
- [X] SeamlessM4T inference *(in development)* https://github.com/facebookresearch/seamless_communication/tree/main/ggml

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
make -j4 gpt-2-backend gpt-j

# Run the GPT-2 small 117M model
../examples/gpt-2/download-ggml-model.sh 117M
./bin/gpt-2-backend -m models/gpt-2-117M/ggml-model.bin -p "This is an example"

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

## Using Metal (only with GPT-2)

For GPT-2 models, offloading to GPU is possible. Note that it will not improve inference performances but will reduce power consumption and free up the CPU for other tasks.

To enable GPU offloading on MacOS:

```bash
cmake -DGGML_METAL=ON -DBUILD_SHARED_LIBS=Off ..

# add -ngl 1
./bin/gpt-2 -t 4 -ngl 100 -m models/gpt-2-117M/ggml-model.bin -p "This is an example"
```

## Using cuBLAS

```bash
# fix the path to point to your CUDA compiler
cmake -DGGML_CUBLAS=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc ..
```

## Using clBLAST

```bash
cmake -DGGML_CLBLAST=ON ..
```
## Compiling for Android

Download and unzip the NDK from this download [page](https://developer.android.com/ndk/downloads). Set the NDK_ROOT_PATH environment variable or provide the absolute path to the CMAKE_ANDROID_NDK in the command below.

```bash
cmake .. \
   -DCMAKE_SYSTEM_NAME=Android \
   -DCMAKE_SYSTEM_VERSION=33 \
   -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
   -DCMAKE_ANDROID_NDK=$NDK_ROOT_PATH
   -DCMAKE_ANDROID_STL_TYPE=c++_shared
```

```bash
# Create directories
adb shell 'mkdir /data/local/tmp/bin'
adb shell 'mkdir /data/local/tmp/models'

# Push the compiled binaries to the folder
adb push bin/* /data/local/tmp/bin/

# Push the ggml library
adb push src/libggml.so /data/local/tmp/

# Push model files
adb push models/gpt-2-117M/ggml-model.bin /data/local/tmp/models/


# Now lets do some inference ...
adb shell

# Now we are in shell
cd /data/local/tmp
export LD_LIBRARY_PATH=/data/local/tmp
./bin/gpt-2-backend -m models/ggml-model.bin -p "this is an example"
```

### CLBlast for Android

Build CLBlast.

```bash
# In CLBlast/build
$ANDROID_SDK_PATH/cmake/3.22.1/bin/cmake .. \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_SYSTEM_VERSION=33 \
    -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
    -DCMAKE_ANDROID_NDK=$ANDROID_NDK_PATH \
    -DCMAKE_ANDROID_STL_TYPE=c++_static \
    -DOPENCL_ROOT=$(readlink -f ../../OpenCL-Headers) \
    -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH \
    -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH

# Build libclblast.so
make -j4
```

Pull `libGLES_mali.so` to `libOpenCL.so`.

```bash
# In ggml project root.
mkdir arm64-v8a
adb pull /system/vendor/lib64/egl/libGLES_mali.so arm64-v8a/libOpenCL.so
```

Build ggml with CLBlast.

```bash
# In ggml/build
cd build
$ANDROID_SDK_PATH/cmake/3.22.1/bin/cmake .. \
    -DGGML_CLBLAST=ON \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_SYSTEM_VERSION=33 \
    -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
    -DCMAKE_ANDROID_NDK=$ANDROID_NDK_PATH \
    -DCMAKE_ANDROID_STL_TYPE=c++_shared \
    -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH \
    -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH \
    -DCLBLAST_HOME=$(readlink -f ../../CLBlast) \
    -DOPENCL_LIB=$(readlink -f ../arm64-v8a/libOpenCL.so)

# Run make, adb push, etc.
```

Then in `adb shell`...

```bash
cd /data/local/tmp
export LD_LIBRARY_PATH=/system/vendor/lib64/egl:/data/local/tmp
./bin/gpt-2-backend -m models/ggml-model.bin -n 64 -p "Pepperoni pizza"
```

OpenCL does not have the same level of support in `ggml-backend` as CUDA or Metal. In the `gpt-2-backend` example, OpenCL will only be used for the matrix multiplications when evaluating large prompts.

## Resources

- [GGML - Large Language Models for Everyone](https://github.com/rustformers/llm/blob/main/crates/ggml/README.md): a description of the GGML format provided by the maintainers of the `llm` Rust crate, which provides Rust bindings for GGML
- [marella/ctransformers](https://github.com/marella/ctransformers): Python bindings for GGML models.
- [go-skynet/go-ggml-transformers.cpp](https://github.com/go-skynet/go-ggml-transformers.cpp): Golang bindings for GGML models
- [smspillaz/ggml-gobject](https://github.com/smspillaz/ggml-gobject): GObject-introspectable wrapper for use of GGML on the GNOME platform.
