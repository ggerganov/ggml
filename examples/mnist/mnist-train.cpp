#include "ggml-opt.h"
#include "mnist-common.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

int main(int argc, char ** argv) {
    if (argc != 5 && argc != 6) {
        fprintf(stderr, "Usage: %s mnist-fc mnist-fc-f32.gguf data/MNIST/raw/train-images-idx3-ubyte data/MNIST/raw/train-labels-idx1-ubyte [CPU/CUDA0]\n", argv[0]);
        exit(0);
    }

    // The MNIST model is so small that the overhead from data shuffling is non-negligible, especially with CUDA.
    // With a shard size of 10 this overhead is greatly reduced at the cost of less shuffling (does not seem to have a significant impact).
    // A batch of 500 images then consists of 50 random shards of size 10 instead of 500 random shards of size 1.
    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(MNIST_NINPUT, MNIST_NCLASSES, MNIST_NTRAIN, /*ndata_shard =*/ 10);

    if (!mnist_image_load(argv[3], dataset)) {
        return 1;
    }
    if (!mnist_label_load(argv[4], dataset)) {
        return 1;
    }

    mnist_model model = mnist_model_init_random(argv[1], argc >= 6 ? argv[5] : "", MNIST_NBATCH_LOGICAL, MNIST_NBATCH_PHYSICAL);

    mnist_model_build(model);

    mnist_model_train(model, dataset, /*nepoch =*/ 30, /*val_split =*/ 0.05f);

    mnist_model_save(model, argv[2]);
}
