# MNIST Examples for GGML

This directory contains simple examples of how to use GGML for training and inference using the [MNIST dataset](https://yann.lecun.com/exdb/mnist/).
All commands listed in this README assume the working directory to be `examples/mnist`.
Please note that training in GGML is a work-in-progress and not production ready.

## Obtaining the data

A description of the dataset can be found on [Yann LeCun's website](https://yann.lecun.com/exdb/mnist/).
While it is also in principle possible to download the dataset from this website these downloads are frequently throttled and
it is recommended to use [HuggingFace](https://huggingface.co/datasets/ylecun/mnist) instead.
The dataset will be downloaded automatically when running `mnist-train-fc.py`.

## Fully connected network

For our first example we will train a fully connected network.
To train a fully connected model in PyTorch and save it as a GGUF file, run:

```bash
$ python3 mnist-train-fc.py mnist-fc-f32.gguf

...

Test loss: 0.066377+-0.010468, Test accuracy: 97.94+-0.14%

Model tensors saved to mnist-fc-f32.gguf:
fc1.weight       (500, 784)
fc1.bias         (500,)
fc2.weight       (10, 500)
fc2.bias         (10,)
```

The training script includes an evaluation of the model on the test set.
To evaluate the model on the CPU using GGML, run:

```bash
$ ../../build/bin/mnist-eval mnist-fc-f32.gguf data/MNIST/raw/t10k-images-idx3-ubyte data/MNIST/raw/t10k-labels-idx1-ubyte

________________________________________________________
________________________________________________________
________________________________________________________
________________________________________________________
__________________________________####__________________
______________________________########__________________
__________________________##########____________________
______________________##############____________________
____________________######________####__________________
__________________________________####__________________
__________________________________####__________________
________________________________####____________________
______________________________####______________________
________________________##########______________________
______________________########__####____________________
________________________##__________##__________________
____________________________________##__________________
__________________________________##____________________
__________________________________##____________________
________________________________##______________________
____________________________####________________________
__________##____________######__________________________
__________##############________________________________
________________####____________________________________
________________________________________________________
________________________________________________________
________________________________________________________
________________________________________________________
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
mnist_model: using CUDA0 (NVIDIA GeForce RTX 3090) as primary backend
mnist_model: unsupported operations will be executed on the following fallback backends (in order of priority):
mnist_model:  - CPU (AMD Ryzen 9 5950X 16-Core Processor)
mnist_model_init_from_file: loading model weights from 'mnist-fc-f32.gguf'
mnist_model_init_from_file: model arch is mnist-fc
mnist_model_init_from_file: successfully loaded weights from mnist-fc-f32.gguf
main: loaded model in 109.44 ms
mnist_model_eval: model evaluation on 10000 images took 76.92 ms, 7.69 us/image
main: predicted digit is 3
main: test_loss=0.066379+-0.009101
main: test_acc=97.94+-0.14%
```

In addition to the evaluation on the test set the GGML evaluation also prints a random image from the test set as well as the model prediction for said image.
To train a fully connected model on the CPU using GGML run:

``` bash
$ ../../build/bin/mnist-train mnist-fc mnist-fc-f32.gguf data/MNIST/raw/train-images-idx3-ubyte data/MNIST/raw/train-labels-idx1-ubyte
```

It can then be evaluated with the same binary as above.

## Convolutional network

To train a convolutional network using TensorFlow run:

```bash
$ python3 mnist-train-cnn.py mnist-cnn-f32.gguf

...

Test loss: 0.047947
Test accuracy: 98.46%
GGUF model saved to 'mnist-cnn-f32.gguf'
```

The saved model can be evaluated on the CPU using the `mnist-eval` binary:

```bash
$ ../../build/bin/mnist-eval mnist-fc-f32.gguf data/MNIST/raw/t10k-images-idx3-ubyte data/MNIST/raw/t10k-labels-idx1-ubyte

________________________________________________________
________________________________________________________
________________________________________________________
________________________________________________________
________________________________________________________
______________________________________##________________
______________________________________##________________
______________________________________##________________
____________________________________##__________________
__________________________________####__________________
__________________________________##____________________
________________________________##______________________
______________________________##________________________
____________________________####________________________
____________________________##__________________________
__________________________##____________________________
________________________##______________________________
______________________##________________________________
____________________####________________________________
____________________##__________________________________
__________________##____________________________________
________________##______________________________________
________________________________________________________
________________________________________________________
________________________________________________________
________________________________________________________
________________________________________________________
________________________________________________________
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
mnist_model: using CUDA0 (NVIDIA GeForce RTX 3090) as primary backend
mnist_model: unsupported operations will be executed on the following fallback backends (in order of priority):
mnist_model:  - CPU (AMD Ryzen 9 5950X 16-Core Processor)
mnist_model_init_from_file: loading model weights from 'mnist-cnn-f32.gguf'
mnist_model_init_from_file: model arch is mnist-cnn
mnist_model_init_from_file: successfully loaded weights from mnist-cnn-f32.gguf
main: loaded model in 91.99 ms
mnist_model_eval: model evaluation on 10000 images took 267.61 ms, 26.76 us/image
main: predicted digit is 1
main: test_loss=0.047955+-0.007029
main: test_acc=98.46+-0.12%
```

Like with the fully connected network the convolutional network can also be trained using GGML:

``` bash
$ ../../build/bin/mnist-train mnist-cnn mnist-cnn-f32.gguf data/MNIST/raw/train-images-idx3-ubyte data/MNIST/raw/train-labels-idx1-ubyte
```

As always, the evaluation is done using `mnist-eval` and like with the fully connected network the GGML graph is exported to `mnist-cnn-f32.ggml`.

## Hardware Acceleration

Both the training and evaluation code is agnostic in terms of hardware as long as the corresponding GGML backend has implemented the necessary operations.
A specific backend can be selected by appending the above commands with a backend name.
The compute graphs then schedule the operations to preferentially use the specified backend.
Note that if a backend does not implement some of the necessary operations a CPU fallback is used instead which may result in bad performance.

## Web demo

The evaluation code can be compiled to WebAssembly using [Emscripten](https://emscripten.org/) (may need to re-login to update `$PATH` after installation).
First, copy the GGUF file of either of the trained models to `examples/mnist` and name it `mnist-f32.gguf`.
Copy the test set to `examples/mnist` and name it `t10k-images-idx3-ubyte`.
Symlinking these files will *not* work!
Compile the code like so:

```bash
$ emcc -I../../include -I../../include/ggml -I../../examples ../../src/ggml.c ../../src/ggml-quants.c ../../src/ggml-aarch64.c mnist-common.cpp -o web/mnist.js -s EXPORTED_FUNCTIONS='["_wasm_eval","_wasm_random_digit","_malloc","_free"]' -s EXPORTED_RUNTIME_METHODS='["ccall"]' -s ALLOW_MEMORY_GROWTH=1 --preload-file mnist-f32.gguf --preload-file t10k-images-idx3-ubyte
```

The compilation output is in `examples/mnist/web`.
To run it, you need an HTTP server.
For example:

``` bash
$ cd web
$ python3 -m http.server

Serving HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...
```

The web demo can then be accessed via the link printed on the console.
Simply draw a digit on the canvas and the model will try to predict what it's supposed to be.
Alternatively, click the "Random" button to retrieve a random digit from the test set.
Be aware that like all neural networks the one we trained is susceptible to distributional shift:
if the numbers you draw look different than the ones in the training set
(e.g. because they're not centered) the model will perform comparatively worse.
An online demo can be accessed [here](https://mnist.ggerganov.com).
