# gpt-j

Local GPT-J inference on your computer using C/C++

No video card required. You just need to have 16 GB of RAM.

For example, you can run this on a 16 GB MacBook M1.

## Motivation

The GPT-J 6B model is the open-source alternative to OpenAI's GPT-3. It's basically a neural network that
allows you to generate coherent, human-like text given a certain context (prompt).

The GPT-J model is quite big - the compact version of the model uses 16-bit floating point representation
of the weights and is still 12 GB big. This means that in order to run inference on your computer, you
would need to have a video card with at least 12 GB of video RAM. Alternatively, you can try to run the
python implementations on the CPU, but that would probably not be very efficient as they are primarily
optimized for running on a GPU (or at least this is my guess - I don't have much experience with python).

Looking on the internet, I couldn't find a dedicated CPU implementation that would allow me to run the model
without a high-end video card. So I decided to write my own inference using a custom build tensor library.
The tensor library (called [ggml](https://github.com/ggerganov/ggml), written in C) is in early development
stage, but it already allows me to run the GPT-J model.

On my MacBook M1 Pro, I achieve an inference speed of about `125 ms/token` or about 2-3 words per second.

Here is a sample run with prompt `int main(int argc, char ** argv) {`:

```
$ time ./bin/gpt-j -p "int main(int argc, char ** argv) {"

gptj_model_load: loading model from 'models/gpt-j-6B/ggml-model.bin' - please wait ...
gptj_model_load: n_vocab = 50400
gptj_model_load: n_ctx   = 2048
gptj_model_load: n_embd  = 4096
gptj_model_load: n_head  = 16
gptj_model_load: n_layer = 28
gptj_model_load: n_rot   = 64
gptj_model_load: f16     = 1
gptj_model_load: ggml ctx size = 13334.86 MB
gptj_model_load: memory_size =  1792.00 MB, n_mem = 57344
gptj_model_load: ................................... done
gptj_model_load: model size = 11542.79 MB / num tensors = 285
main: number of tokens in prompt = 13

int main(int argc, char ** argv) {
    (void)argc;
    (void)argv;

    {
        struct sockaddr_in addr;
        int addrlen;
        char * ip = "192.168.1.4";
        int i;

        if ( (addrlen = sizeof(addr)) == -1 )
            return -1;

        for (i = 0; i < 10; ++i) {
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = inet_addr(ip);

main: mem per token = 16430420 bytes
main:     load time =  6211.48 ms
main:   sample time =    13.74 ms
main:  predict time = 26420.34 ms / 124.62 ms per token
main:    total time = 33035.37 ms

real	0m33.171s
user	3m32.269s
sys	     0m3.686s

$
```

It took ~6.2 seconds to load the model to memory. After that, it took ~26.4 seconds to generate 200
tokens of what looks like to be the beginning of a networking program in C. Pretty cool!

## Implementation details

The high level implementation of the model is contained in the [main.cpp](main.cpp) file. The core
computations are performed by the `ggml` library.

The most performance critical part of the implementation is of course the matrix multiplication routine.
99% of the time is spent here, so it is important to optimize this as much as possible.

On Arm64, I utilize the 128-bit NEON intrinsics for 16-bit floating point operations:

https://github.com/ggerganov/ggml/blob/fb558f78d905f85c54813602649ddd628ffe0f3a/src/ggml.c#L187-L243

These instructions allow each core to operate simultaneously on 64 floating point numbers. I'm no expert
in SIMD, but after quite some trials this was the most efficient code for dot product that I could come up
with. Combined with the parallel computation on 8 CPU threads, I think I got close to the maximum performance
that one could possibly get on the M1 CPU. Still, I'm curious to know if there is a more efficient way to
implement this.

One interesting property of the GPT-J transformer architecture is that it allows you to perform part
of the inference in parallel - i.e. the Feed-forward layer can be computed in parallel to the Self-Attention
layer:

https://github.com/ggerganov/ggml/blob/fb558f78d905f85c54813602649ddd628ffe0f3a/examples/gpt-j/main.cpp#L507-L531

So I thought why not bring in the M1 GPU to compute half of the neural network in parallel to the CPU.
Thanks to the shared memory model, it was relatively easy to offload half of the computation to the GPU
using [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders).
However, to my surprise, I did not get any performance improvement at all. My conclusion was that the
8-thread NEON CPU computation is basically saturating the memory bandwidth of the M1 and since the CPU
and the GPU on the MacBook are sharing that bandwidth, it does not help to offload the computation to the
GPU. Another observation was that the MPS GPU matrix multiplication using 16-bit floats had the same
performance as the 8-thread NEON CPU implementation. Again, I explain this with a saturated memory channel.
But of course, I could be totally wrong and somehow my implementation wasn't utilizing the resources 
correctly.

Another property of my implementation is that it does not perform any memory allocations once the model
is loaded into memory. All required memory is allocated at the start of the program.

## Usage

If you want to give this a try and you are on Linux or Mac OS, simply follow these instructions:

```bash
# Clone the ggml library and build the gpt-j example
git clone https://github.com/ggerganov/ggml
cd ggml
mkdir build && cd build
cmake ..
make -j4 gpt-j

# Download the ggml-compatible GPT-J 6B model (requires 12GB disk space)
../examples/gpt-j/download-ggml-model.sh 6B

# Run the inference (requires 16GB of CPU RAM)
./bin/gpt-j -m models/gpt-j-6B/ggml-model.bin -p "This is an example"
```

To run the `gpt-j` tool, you need the 12GB `ggml-model.bin` file which contains the GPT-J model in
[ggml](https://github.com/ggerganov/ggml) format. In the instructions above, I download the binary file
directly from one of my servers, using the [download-ggml-model.sh](download-ggml-model.sh) script.

---

Alternatively, you can perform the conversion yourself.

First, you need to download the full GPT-J model from here: https://huggingface.co/EleutherAI/gpt-j-6B

Note that the full model is quite big - about 72 GB. After you download it, you need to make the
conversion using the [convert-h5-to-ggml.py](convert-h5-to-ggml.py) script. This will generate the
`ggml-model.bin` file, which you can then use with the `gpt-j` program.

## GPT-2

I have also implemented a tool for CPU inference using the smaller GPT-2 models. They have worse
quality compared to GPT-J, but are much faster to execute.

Checkout the GPT-2 example here: [gpt-2](https://github.com/ggerganov/ggml/tree/master/examples/gpt-2)
