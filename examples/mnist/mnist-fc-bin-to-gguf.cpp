#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>

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
bool mnist_model_convert(const std::string & fname_in, const std::string & fname_out) {
    printf("%s: loading model from '%s'\n", __func__, fname_in.c_str());

    auto fin = std::ifstream(fname_in, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname_in.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname_in.c_str());
            return false;
        }
    }

    mnist_model model;
    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_input   = hparams.n_input;
        const int n_hidden  = hparams.n_hidden;
        const int n_classes = hparams.n_classes;

        ctx_size += n_input * n_hidden * ggml_type_size(GGML_TYPE_F32); // fc1 weight
        ctx_size +=           n_hidden * ggml_type_size(GGML_TYPE_F32); // fc1 bias

        ctx_size += n_hidden * n_classes * ggml_type_size(GGML_TYPE_F32); // fc2 weight
        ctx_size +=            n_classes * ggml_type_size(GGML_TYPE_F32); // fc2 bias

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size + 1024*1024,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
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

        {
            int32_t ne_weight[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne_weight[i]), sizeof(ne_weight[i]));
            }

            // FC1 dimensions taken from file, eg. 768x500
            model.hparams.n_input  = ne_weight[0];
            model.hparams.n_hidden = ne_weight[1];

            model.fc1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.hparams.n_input, model.hparams.n_hidden);
            fin.read(reinterpret_cast<char *>(model.fc1_weight->data), ggml_nbytes(model.fc1_weight));
            ggml_set_name(model.fc1_weight, "fc1_weight");
        }

        {
            int32_t ne_bias[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne_bias[i]), sizeof(ne_bias[i]));
            }

            model.fc1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_hidden);
            fin.read(reinterpret_cast<char *>(model.fc1_bias->data), ggml_nbytes(model.fc1_bias));
            ggml_set_name(model.fc1_bias, "fc1_bias");

            // just for testing purposes, set some parameters to non-zero
            model.fc1_bias->op_params[0] = 0xdeadbeef;
        }
    }

    // Read FC2 layer 2
    {
        // Read dimensions
        int32_t n_dims;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));

        {
            int32_t ne_weight[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne_weight[i]), sizeof(ne_weight[i]));
            }

            // FC1 dimensions taken from file, eg. 10x500
            model.hparams.n_classes = ne_weight[1];

            model.fc2_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.hparams.n_hidden, model.hparams.n_classes);
            fin.read(reinterpret_cast<char *>(model.fc2_weight->data), ggml_nbytes(model.fc2_weight));
            ggml_set_name(model.fc2_weight, "fc2_weight");
        }

        {
            int32_t ne_bias[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne_bias[i]), sizeof(ne_bias[i]));
            }

            model.fc2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_classes);
            fin.read(reinterpret_cast<char *>(model.fc2_bias->data), ggml_nbytes(model.fc2_bias));
            ggml_set_name(model.fc2_bias, "fc2_bias");
        }
    }

    fin.close();

    gguf_context * gguf_ctx = gguf_init_empty();
    gguf_add_tensor(gguf_ctx, model.fc1_weight);
    gguf_add_tensor(gguf_ctx, model.fc1_bias);
    gguf_add_tensor(gguf_ctx, model.fc2_weight);
    gguf_add_tensor(gguf_ctx, model.fc2_bias);
    gguf_write_to_file(gguf_ctx, fname_out.c_str(), false);

    return true;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s models/mnist/ggml-model-f32.bin models/mnist/ggml-model-f32.gguf\n", argv[0]);
        exit(0);
    }

    if (!mnist_model_convert(argv[1], argv[2])) {
        fprintf(stderr, "%s: failed to convert model from '%s' to '%s'\n", __func__, argv[1], argv[2]);
        return 1;
    }

    return 0;
}
