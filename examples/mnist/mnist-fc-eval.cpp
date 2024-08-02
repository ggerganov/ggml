#include "ggml.h"

#include "mnist-common.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#ifdef __cplusplus
extern "C" {
#endif

int wasm_eval(uint8_t * digitPtr) {
    mnist_model model;
    // FIXME
    // if (!mnist_model_load("models/mnist/ggml-model-f32.bin", model)) {
    //     fprintf(stderr, "error loading model\n");
    //     return -1;
    // }
    std::vector<float> digit(digitPtr, digitPtr + 784);
    // int result = mnist_eval(model, 1, digit, nullptr); // FIXME
    int result = -1;
    ggml_free(model.ctx_gguf); // FIXME

    return result;
}

int wasm_random_digit(char * digitPtr) {
    auto fin = std::ifstream("models/mnist/t10k-images.idx3-ubyte", std::ios::binary);
    if (!fin) {
        fprintf(stderr, "failed to open digits file\n");
        return 0;
    }
    srand(time(NULL));

    // Seek to a random digit: 16-byte header + 28*28 * (random 0 - 10000)
    fin.seekg(16 + 784 * (rand() % 10000));
    fin.read(digitPtr, 784);

    return 1;
}

#ifdef __cplusplus
}
#endif

int main(int argc, char ** argv) {
    srand(time(NULL));
    ggml_time_init();

    if (argc != 4) {
        fprintf(stderr, "Usage: %s models/MNIST/mnist-fc-f32.gguf data/MNIST/raw/t10k-images-idx3-ubyte data/MNIST/raw/t10k-labels-idx1-ubyte\n", argv[0]);
        exit(0);
    }

    mnist_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        model = mnist_model_init(argv[1], 1);

        const int64_t t_load_us = ggml_time_us() - t_start_us;

        fprintf(stdout, "%s: loaded model in %8.2f ms\n", __func__, t_load_us / 1000.0f);
    }

    std::vector<float> images;
    images.resize(MNIST_NTEST*MNIST_NINPUT);
    if (!mnist_image_load(argv[2], images.data(), MNIST_NTEST, false)) {
        return 1;
    }

    std::vector<float> labels;
    labels.resize(MNIST_NTEST*MNIST_NCLASSES);
    if (!mnist_label_load(argv[3], labels.data(), MNIST_NTEST)) {
        return 1;
    }

    const int iex = rand() % MNIST_NTEST;
    const std::vector<float> digit(images.begin() + iex*MNIST_NINPUT, images.begin() + (iex+1)*MNIST_NINPUT);

    mnist_image_print(stdout, images.data() + iex*MNIST_NINPUT);

    const mnist_eval_result result_eval = mnist_model_eval(model, images.data(), labels.data(), MNIST_NTEST);
    fprintf(stdout, "%s: predicted digit is %d\n", __func__, result_eval.pred[iex]);

    std::pair<double, double> result_acc = mnist_accuracy(result_eval, labels.data());
    fprintf(stdout, "%s: test_acc=%.2f+-%.2f%%\n", __func__, 100.0*result_acc.first, 100.0*result_acc.second);

    mnist_model_free(model);

    return 0;
}
