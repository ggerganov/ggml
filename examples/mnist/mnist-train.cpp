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
    if (argc != 5) {
        fprintf(stderr, "Usage: %s mnist-fc models/MNIST/mnist-fc-f32.gguf data/MNIST/raw/train-images-idx3-ubyte data/MNIST/raw/train-labels-idx1-ubyte\n", argv[0]);
        exit(0);
    }

    std::vector<float> images;
    images.resize(MNIST_NTRAIN*MNIST_NINPUT);
    if (!mnist_image_load(argv[3], images.data(), MNIST_NTRAIN)) {
        return 1;
    }

    std::vector<float> labels;
    labels.resize(MNIST_NTRAIN*MNIST_NCLASSES);
    if (!mnist_label_load(argv[4], labels.data(), MNIST_NTRAIN)) {
        return 1;
    }

    const int nex = MNIST_NTRAIN;
    mnist_model model = mnist_model_init_random(argv[1]);

    mnist_model_build(model);

    mnist_model_train(images.data(), labels.data(), nex, model);

    mnist_model_save(argv[2], model);
}
