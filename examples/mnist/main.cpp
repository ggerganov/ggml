#include "ggml.h"

#include "common.h"

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

// default hparams
struct mnist_hparams {
    int32_t n_input   = 784;
    int32_t n_hidden  = 500;
    int32_t n_classes = 10;
};

struct mnist_model {
    mnist_hparams hparams;

    struct ggml_tensor * fc1_weight;
    struct ggml_tensor * fc1_bias;

    struct ggml_tensor * fc2_weight;
    struct ggml_tensor * fc2_bias;

    struct ggml_context * ctx;
};

// load the model's weights from a file
bool mnist_model_load(const std::string & fname, mnist_model & model) {
    printf("%s: loading model from '%s'\n", __func__, fname.c_str());

    struct gguf_init_params params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &model.ctx,
    };
    gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);
    if (!ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }
    model.fc1_weight = ggml_get_tensor(model.ctx, "fc1_weight");
    model.fc1_bias   = ggml_get_tensor(model.ctx, "fc1_bias");
    model.fc2_weight = ggml_get_tensor(model.ctx, "fc2_weight");
    model.fc2_bias   = ggml_get_tensor(model.ctx, "fc2_bias");

    return true;
}

// evaluate the model
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - digit:     784 pixel values
//
// returns 0 - 9 prediction
int mnist_eval(
        const mnist_model & model,
        const int n_threads,
        std::vector<float> digit,
        const char * fname_cgraph
        ) {

    const auto & hparams = model.hparams;

    static size_t buf_size = hparams.n_input * sizeof(float) * 32;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hparams.n_input);
    memcpy(input->data, digit.data(), ggml_nbytes(input));
    ggml_set_name(input, "input");

    // fc1 MLP = Ax + b
    ggml_tensor * fc1 = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc1_weight, input),                model.fc1_bias);
    ggml_tensor * fc2 = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc2_weight, ggml_relu(ctx0, fc1)), model.fc2_bias);

    // soft max
    ggml_tensor * probs = ggml_soft_max(ctx0, fc2);
    ggml_set_name(probs, "probs");

    // build / export / run the computation graph
    ggml_build_forward_expand(gf, probs);
    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

    //ggml_graph_print   (&gf);
    ggml_graph_dump_dot(gf, NULL, "mnist.dot");

    if (fname_cgraph) {
        // export the compute graph for later use
        // see the "mnist-cpu" example
        ggml_graph_export(gf, "mnist.ggml");

        fprintf(stderr, "%s: exported compute graph to '%s'\n", __func__, fname_cgraph);
    }

    const float * probs_data = ggml_get_data_f32(probs);

    const int prediction = std::max_element(probs_data, probs_data + 10) - probs_data;

    ggml_free(ctx0);

    return prediction;
}

#ifdef __cplusplus
extern "C" {
#endif

int wasm_eval(uint8_t * digitPtr) {
    mnist_model model;
    if (!mnist_model_load("models/mnist/ggml-model-f32.bin", model)) {
        fprintf(stderr, "error loading model\n");
        return -1;
    }
    std::vector<float> digit(digitPtr, digitPtr + 784);
    int result = mnist_eval(model, 1, digit, nullptr);
    ggml_free(model.ctx);

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

    if (argc != 3) {
        fprintf(stderr, "Usage: %s models/mnist/ggml-model-f32.bin models/mnist/t10k-images.idx3-ubyte\n", argv[0]);
        exit(0);
    }

    uint8_t buf[784];
    mnist_model model;
    std::vector<float> digit;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!mnist_model_load(argv[1], model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, argv[1]);
            return 1;
        }

        const int64_t t_load_us = ggml_time_us() - t_start_us;

        fprintf(stdout, "%s: loaded model in %8.2f ms\n", __func__, t_load_us / 1000.0f);
    }

    // read a random digit from the test set
    {
        std::ifstream fin(argv[2], std::ios::binary);
        if (!fin) {
            fprintf(stderr, "%s: failed to open '%s'\n", __func__, argv[2]);
            return 1;
        }

        // seek to a random digit: 16-byte header + 28*28 * (random 0 - 10000)
        fin.seekg(16 + 784 * (rand() % 10000));
        fin.read((char *) &buf, sizeof(buf));
    }

    // render the digit in ASCII
    {
        digit.resize(sizeof(buf));

        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                fprintf(stderr, "%c ", (float)buf[row*28 + col] > 230 ? '*' : '_');
                digit[row*28 + col] = ((float)buf[row*28 + col]);
            }

            fprintf(stderr, "\n");
        }

        fprintf(stderr, "\n");
    }

    const int prediction = mnist_eval(model, 1, digit, "mnist.ggml");

    fprintf(stdout, "%s: predicted digit is %d\n", __func__, prediction);

    ggml_free(model.ctx);

    return 0;
}
