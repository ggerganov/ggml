#include "ggml/ggml.h"

#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <unistd.h>
#include <time.h>
#include <algorithm>

// default hparams
struct mnist_hparams {
    int32_t n_input = 784;
    int32_t n_hidden = 500;
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

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    const ggml_type wtype2 = GGML_TYPE_F32;

    auto & ctx = model.ctx;
    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_input  = hparams.n_input;
        const int n_hidden = hparams.n_hidden;
        const int n_classes = hparams.n_classes;

        // fc1 weight
        ctx_size += n_input * n_hidden * ggml_type_sizef(GGML_TYPE_F32);
        // fc1 bias
        ctx_size += n_hidden * ggml_type_sizef(GGML_TYPE_F32);

        // fc2 weight
        ctx_size += n_hidden * n_classes * ggml_type_sizef(GGML_TYPE_F32);
        // fc2 bias
        ctx_size += n_classes * ggml_type_sizef(GGML_TYPE_F32);

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            .mem_size   = ctx_size + 1024*1024,
            .mem_buffer = NULL,
            .no_alloc   = false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // Read FC1 layer 1
    {
        // Read dimensions
        int32_t n_dims;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));

        int32_t ne_weight[2] = { 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            fin.read(reinterpret_cast<char *>(&ne_weight[i]), sizeof(ne_weight[i]));
        }

        // FC1 dimensions taken from file, eg. 768x500
        model.hparams.n_input = ne_weight[0];
        model.hparams.n_hidden = ne_weight[1];

        model.fc1_weight     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,   model.hparams.n_input, model.hparams.n_hidden);
        fin.read(reinterpret_cast<char *>(model.fc1_weight->data), ggml_nbytes(model.fc1_weight));

        int32_t ne_bias[2] = { 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            fin.read(reinterpret_cast<char *>(&ne_bias[i]), sizeof(ne_bias[i]));
        }

        model.fc1_bias    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_hidden);
        fin.read(reinterpret_cast<char *>(model.fc1_bias->data), ggml_nbytes(model.fc1_bias));
    }

    // Read FC2 layer 2
    {
        // Read dimensions
        int32_t n_dims;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));

        int32_t ne_weight[2] = { 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            fin.read(reinterpret_cast<char *>(&ne_weight[i]), sizeof(ne_weight[i]));
        }

        // FC1 dimensions taken from file, eg. 10x500
        model.hparams.n_classes = ne_weight[1];

        model.fc2_weight     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.hparams.n_hidden, model.hparams.n_classes);
        fin.read(reinterpret_cast<char *>(model.fc2_weight->data), ggml_nbytes(model.fc2_weight));

        int32_t ne_bias[2] = { 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            fin.read(reinterpret_cast<char *>(&ne_bias[i]), sizeof(ne_bias[i]));
        }
        model.fc2_bias     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_classes);
        fin.read(reinterpret_cast<char *>(model.fc2_bias->data), ggml_nbytes(model.fc2_bias));
    }
    fin.close();

    return true;
}

// evaluate the model
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - digit:   784 pixel values
// returns 0 - 9 prediction
int mnist_eval(
        const mnist_model & model,
        const int n_threads,
        std::vector<float> digit
        ) {

    const auto & hparams = model.hparams;

    static size_t buf_size = hparams.n_input * sizeof(float) * 4;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = buf,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = { .n_threads = n_threads };

    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hparams.n_input);
    memcpy(input->data, digit.data(), ggml_nbytes(input));

    // fc1 MLP = Ax + b
    ggml_tensor * fc1 = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc1_weight, input), model.fc1_bias);
    ggml_tensor * fc2 = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc2_weight, ggml_relu(ctx0, fc1)), model.fc2_bias);

    // soft max
    ggml_tensor * final = ggml_soft_max(ctx0, fc2);

    // run the computation
    ggml_build_forward_expand(&gf, final);
    ggml_graph_compute       (ctx0, &gf);

    //ggml_graph_print   (&gf);
    ggml_graph_dump_dot(&gf, NULL, "mnist.dot");
    float* finalData = ggml_get_data_f32(final);

    int prediction = std::max_element(finalData, finalData + 10) - finalData;
    ggml_free(ctx0);
    return prediction;
}

int main(int argc, char ** argv) {

    if (argc != 3) {
        fprintf(stderr, "Usage: %s models/mnist/ggml-model-f32.bin models/mnist/t10k-images.idx3-ubyte\n", argv[0]);
        exit(0);
    }
    const int64_t t_main_start_us = ggml_time_us();

    mnist_hparams params;
    int64_t t_load_us = 0;

    mnist_model model;
    std::vector<float> digit;
    // load the model, load a random test digit, evaluate the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!mnist_model_load(argv[1], model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, "models/ggml-model-f32.bin");
            return 1;
        }
        auto fin = std::ifstream(argv[2], std::ios::binary);
        if (!fin) {
            fprintf(stderr, "%s: failed to open '%s'\n", __func__, argv[2]);
            return 1;
        }

        unsigned char buf[784];
        srand(time(NULL));
        // Seek to a random digit: 16-byte header + 28*28 * (random 0 - 10000)
        fin.seekg(16 + 784 * (rand() % 10000));
        fin.read((char *) &buf, sizeof(buf));
        digit.resize(sizeof(buf));

        // render the digit in ASCII
        for(int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                fprintf(stderr, "%c ", (float)buf[row*28 + col] > 230 ? '*' : '_');
                digit[row*28+col]=((float)buf[row*28+col]);

            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");

        t_load_us = ggml_time_us() - t_start_us;
    }


    fprintf(stdout, "Predicted digit is %d\n", mnist_eval(model, 1, digit));
    ggml_free(model.ctx);

    return 0;
}
