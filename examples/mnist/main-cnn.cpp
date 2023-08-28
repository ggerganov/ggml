#include "ggml/ggml.h"

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

struct mnist_model {
    struct ggml_tensor * conv2d_1_kernel;
    struct ggml_tensor * conv2d_1_bias;
    struct ggml_tensor * conv2d_2_kernel;
    struct ggml_tensor * conv2d_2_bias;
    struct ggml_tensor * dense_weight;
    struct ggml_tensor * dense_bias;
    struct ggml_context * ctx;
};

bool mnist_model_load(const std::string & fname, mnist_model & model) {
    struct gguf_init_params params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &model.ctx,
    };
    gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);
    if (!ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }
    model.conv2d_1_kernel = ggml_get_tensor(model.ctx, "kernel1");
    model.conv2d_1_bias = ggml_get_tensor(model.ctx, "bias1");
    model.conv2d_2_kernel = ggml_get_tensor(model.ctx, "kernel2");
    model.conv2d_2_bias = ggml_get_tensor(model.ctx, "bias2");
    model.dense_weight = ggml_get_tensor(model.ctx, "dense_w");
    model.dense_bias = ggml_get_tensor(model.ctx, "dense_b");
    return true;
}

int mnist_eval(
        const mnist_model & model,
        const int n_threads,
        std::vector<float> digit,
        const char * fname_cgraph
        )
{
    static size_t buf_size = 100000 * sizeof(float) * 4;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};

    struct ggml_tensor * input = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 28, 28, 1, 1);
    memcpy(input->data, digit.data(), ggml_nbytes(input));
    ggml_set_name(input, "input");
    ggml_tensor * cur = ggml_conv_2d(ctx0, model.conv2d_1_kernel, input, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, model.conv2d_1_bias);
    cur = ggml_relu(ctx0, cur);
    // Output shape after Conv2D: (26 26 32 1)
    cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    // Output shape after MaxPooling2D: (13 13 32 1)
    cur = ggml_conv_2d(ctx0, model.conv2d_2_kernel, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, model.conv2d_2_bias);
    cur = ggml_relu(ctx0, cur);
    // Output shape after Conv2D: (11 11 64 1)
    cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    // Output shape after MaxPooling2D: (5 5 64 1)
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));
    // Output shape after permute: (64 5 5 1)
    cur = ggml_reshape_2d(ctx0, cur, 1600, 1);
    // Final Dense layer
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.dense_weight, cur), model.dense_bias);
    ggml_tensor * probs = ggml_soft_max(ctx0, cur);
    ggml_set_name(probs, "probs");

    ggml_build_forward_expand(&gf, probs);
    ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

    //ggml_graph_print(&gf);
    ggml_graph_dump_dot(&gf, NULL, "mnist-cnn.dot");

    if (fname_cgraph) {
        // export the compute graph for later use
        // see the "mnist-cpu" example
        ggml_graph_export(&gf, fname_cgraph);

        fprintf(stderr, "%s: exported compute graph to '%s'\n", __func__, fname_cgraph);
    }

    const float * probs_data = ggml_get_data_f32(probs);
    const int prediction = std::max_element(probs_data, probs_data + 10) - probs_data;
    ggml_free(ctx0);
    return prediction;
}

int main(int argc, char ** argv) {
    srand(time(NULL));
    ggml_time_init();

    if (argc != 3) {
        fprintf(stderr, "Usage: %s models/mnist/mnist-cnn.gguf models/mnist/t10k-images.idx3-ubyte\n", argv[0]);
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
                digit[row*28 + col] = ((float)buf[row*28 + col] / 255.0f);
            }

            fprintf(stderr, "\n");
        }

        fprintf(stderr, "\n");
    }

    const int prediction = mnist_eval(model, 1, digit, nullptr);
    fprintf(stdout, "%s: predicted digit is %d\n", __func__, prediction);
    ggml_free(model.ctx);
    return 0;
}
