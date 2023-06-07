# MNIST Example for GGML

This is a simple example of how to use GGML for inferencing.

## Training the Model

A Google Colab notebook for training a simple two-layer network to recognize digits is located here. You can
use this to save a pytorch model to be converted to ggml format.

[Colab](https://colab.research.google.com/drive/12n_8VNJnolBnX5dVS0HNWubnOjyEaFSb?usp=sharing)


## GGML Format Conversion

GGML "format" is whatever you choose for efficient loading. In our case, we just save the hyperparameters used
plus the model weights and biases. Run convert-h5-to-ggml.py to convert your pytorch model. The output format is:

- magic constant (int32)
- repeated list of tensors
- number of dimensions of tensor (int32)
- tensor dimension (int32 repeated)
- values of tensor (int32)

Run ```convert-h5-to-ggml.py mnist_model.state_dict``` where `mnist_model.state_dict` is the saved pytorch model from the Google Colab. For
quickstart, it is included in the mnist/models directory.

## MNIST Network

The MNIST recognizer network is extremely simple. A fully connected layer + relu, followed by a fully connected layer + softmax. This
version of the MNIST network doesn't use convolutions.

## Running the example

Here is how to run the example programs:

```bash
# Build ggml + examples
git clone https://github.com/ggerganov/ggml
cd ggml
mkdir build && cd build
cmake ..
make -j4 mnist

# Run the MNIST model
./bin/mnist ../examples/mnist/models/mnist/ggml-model-f32.bin ../examples/mnist/models/mnist/t10k-images.idx3-ubyte
```

For more information, checkout the corresponding programs in the [examples](examples) folder.

# Sample output


```
$  ./bin/mnist ./models/mnist/ggml-model-f32.bin ../examples/mnist/models/mnist/t10k-images.idx3-ubyte
mnist_model_load: loading model from './models/mnist/ggml-model-f32.bin'
mnist_model_load: ggml ctx size =   1.52 MB
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ * * _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ * * * * _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ * * * _ _ _ * _ * * _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ * * _ _ _ _ _ * _ * _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ * * _ _ _ _ _ _ * * _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ * * _ _ _ _ _ _ * * * * _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ * * _ _ _ _ _ _ _ * * * * _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ * * _ _ _ _ _ * * _ _ * * _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ * * _ _ _ _ * * _ _ _ _ * _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ * * * * * * _ _ _ _ _ _ * _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * * _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * * _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * * _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * * _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ * * * _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ * _ _ * * * _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ * * * * * _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

ggml_graph_dump_dot: dot -Tpng mnist.dot -o mnist.dot.png && open mnist.dot.png
Predicted digit is 9
```

Computation graph:

![mnist dot](https://user-images.githubusercontent.com/1991296/231882071-84e29d53-b226-4d73-bdc2-5bd6dcb7efd1.png)


## Web demo

The example can be compiled with Emscripten like this:

```bash
cd examples/mnist
emcc -I../../include -I../../include/ggml -I../../examples ../../src/ggml.c main.cpp -o web/mnist.js -s EXPORTED_FUNCTIONS='["_wasm_eval","_wasm_random_digit","_malloc","_free"]' -s EXPORTED_RUNTIME_METHODS='["ccall"]' -s ALLOW_MEMORY_GROWTH=1 --preload-file models/mnist
```

Online demo: https://mnist.ggerganov.com
