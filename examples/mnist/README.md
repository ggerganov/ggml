# MNIST Examples for GGML

This directory contains simple examples of how to use GGML for training and inference using the [MNIST dataset](https://yann.lecun.com/exdb/mnist/).
All commands listed in this README assume the working directory to be `examples/mnist`.
Please note that training in GGML is a work-in-progress and not production ready.

## Obtaining the data

The data can either be downloaded [here](https://yann.lecun.com/exdb/mnist/) or it will be downloaded automatically when running `mnist-train-fc.py`.

## Fully connected network

For our first example we will train a fully connected network.
To train a fully connected model in PyTorch and save it as a GGUF file, run:

```bash
$ python3 mnist-train-fc.py mnist-fc-f32.gguf

...

Test loss: 0.069983+-0.009196, Test accuracy: 97.94+-0.14%

Model tensors saved to mnist-fc-f32.gguf:
fc1.weight       (500, 784)
fc1.bias         (500,)
fc2.weight       (10, 500)
fc2.bias         (10,)
```

The training script includes an evaluation of the model on the test set.
To evaluate the model using GGML, run:

```bash
$ ../../build/bin/mnist-eval mnist-fc-f32.gguf data/MNIST/raw/t10k-images-idx3-ubyte data/MNIST/raw/t10k-labels-idx1-ubyte

________________________________________________________
________________________________________________________
________________________________________________________
________________________________________________________
________________________________######__________________
____________________________########____________________
________________________########________________________
____________________########________________##__________
__________________######____________________##__________
________________######______________________####________
______________######________________________####________
____________######__________________________####________
____________####____________________________####________
__________####______________________________####________
__________####______________________________####________
__________##________________________________####________
__________##______________________________####__________
__________##____________________________######__________
__________##__________________________######____________
____________##____________________########______________
____________##########################__________________
______________##################________________________
________________________________________________________
________________________________________________________
________________________________________________________
________________________________________________________
________________________________________________________
________________________________________________________
mnist_graph_eval: trying to load a ggml graph from mnist-fc-f32.gguf
ggml_graph_import: invalid magic number, got 46554747
mnist_graph_eval: could not load a ggml graph from mnist-fc-f32.gguf
mnist_model_init_from_file: loading model weights from 'mnist-fc-f32.gguf'
mnist_model_init_from_file: model arch is mnist-fc
mnist_model_init_from_file: successfully loaded weights from mnist-fc-f32.gguf
main: loaded model in 1.52 ms
mnist_model_eval: model evaluation on 10000 images took 26.65 ms, 2.66 us/image
main: predicted digit is 0
main: test_loss=0.069983+-0.009196
main: test_acc=97.94+-0.14%
```

In addition to the evaluation on the test set the GGML evaluation also prints a random image from the test set as well as the model prediction for said image.
To train a fully connected model using GGML run:

``` bash
$ ../../build/bin/mnist-train mnist-fc mnist-fc-f32.gguf data/MNIST/raw/train-images-idx3-ubyte data/MNIST/raw/train-labels-idx1-ubyte
```

It can then be evaluated with the same binary as above.
When training a model with GGML the computation graph for the forward pass is also exported to `mnist-fc-f32.ggml`.
Compared to the GGUF (which only contains the weights) this file also contains the model architecture.
As long as the input and output tensors are well-defined an exported GGML graph is fully agnostic w.r.t. the model architecture.
It can be evaluated using the `mnist-eval` binary by substituting the argument for the GGUF file.

## Convolutional network

To train a convolutional network using TensorFlow run:

```bash
$ python3 mnist-train-cnn.py mnist-cnn-f32.gguf

...

Test loss: 0.046456
Test accuracy: 98.40%
GGUF model saved to 'mnist-cnn-f32.gguf'
```

The saved model can be evaluated using the `mnist-eval` binary:

```bash
$ ../../build/bin/mnist-eval mnist-fc-f32.gguf data/MNIST/raw/t10k-images-idx3-ubyte data/MNIST/raw/t10k-labels-idx1-ubyte

________________________________________________________
________________________________________________________
________________________________________________________
________________________________________________________
________________________________________________________
________________________________________________________
________________________________________________________
________________________####____________________________
__________________________##____________________________
__________________________##____________________________
__________________________##____________________________
__________________________##____________________________
__________________________##____________________________
____________________________##__________________________
____________________________##__________________________
____________________________##__________________________
______________________________##________________________
______________________________##________________________
______________________________####______________________
________________________________##______________________
________________________________##______________________
________________________________####____________________
__________________________________##____________________
________________________________##______________________
________________________________________________________
________________________________________________________
________________________________________________________
________________________________________________________
mnist_graph_eval: trying to load a ggml graph from mnist-cnn-f32.gguf
ggml_graph_import: invalid magic number, got 46554747
mnist_graph_eval: could not load a ggml graph from mnist-cnn-f32.gguf
mnist_model_init_from_file: loading model weights from 'mnist-cnn-f32.gguf'
mnist_model_init_from_file: model arch is mnist-cnn
mnist_model_init_from_file: successfully loaded weights from mnist-cnn-f32.gguf
main: loaded model in 5.45 ms
mnist_model_eval: model evaluation on 10000 images took 605.60 ms, 60.56 us/image
main: predicted digit is 1
main: test_loss=0.046456+-0.007354
main: test_acc=98.40+-0.13%
```

Like with the fully connected network the convolutional network can also be trained using GGML:

``` bash
$ ../../build/bin/mnist-train mnist-cnn mnist-cnn-f32.gguf data/MNIST/raw/train-images-idx3-ubyte data/MNIST/raw/train-labels-idx1-ubyte
```

As always, the evaluation is done using `mnist-eval` and like with the fully connected network the GGML graph is exported to `mnist-cnn-f32.ggml`.

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
