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
    if (argc != 4) {
        fprintf(stderr, "Usage: %s data/MNIST/raw/train-images-idx3-ubyte data/MNIST/raw/train-labels-idx1-ubyte models/MNIST/mnist-fc-f32.gguf\n", argv[0]);
        exit(0);
    }

    std::vector<float> images;
    images.resize(MNIST_NTRAIN*MNIST_NINPUT);
    if (!mnist_image_load(argv[1], images.data(), MNIST_NTRAIN)) {
        return 1;
    }

    std::vector<float> labels;
    labels.resize(MNIST_NTRAIN*MNIST_NCLASSES);
    if (!mnist_label_load(argv[2], labels.data(), MNIST_NTRAIN)) {
        return 1;
    }

    const int nex    = MNIST_NTRAIN;
    const int nbatch = 1000;
    static_assert(nex % nbatch == 0, "nex % nbatch != 0");

    mnist_model model = mnist_model_init("", nbatch);

    mnist_model_train(images.data(), labels.data(), nex, model);

    mnist_model_save(argv[3], model);

    mnist_model_free(model);
}
