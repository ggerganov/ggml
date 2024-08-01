#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <random>
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

bool mnist_data_load(const std::string & fname_images, const std::string & fname_labels, float * images, float * labels, const int nex) {
    auto fin_images = std::ifstream(fname_images, std::ios::binary);
    if (!fin_images) {
        fprintf(stderr, "failed to open images file %s\n", fname_images.c_str());
        return 0;
    }
    fin_images.seekg(16);

    auto fin_labels = std::ifstream(fname_labels, std::ios::binary);
    if (!fin_labels) {
        fprintf(stderr, "failed to open labels file %s\n", fname_labels.c_str());
        return 0;
    }
    fin_labels.seekg(8);

    uint8_t image[784];
    uint8_t label;

    for (int iex = 0; iex < nex; ++iex) {
        fin_images.read((char *)  image, sizeof(image));
        fin_labels.read((char *) &label, sizeof(label));

        // {
        //     for (int row = 0; row < 28; row++) {
        //         for (int col = 0; col < 28; col++) {
        //             fprintf(stderr, "%c ", image[row*28 + col] > 230 ? '*' : '_');
        //         }
        //         fprintf(stderr, "\n");
        //     }
        //     fprintf(stderr, "\nlabel=%d\n", (int)label);
        // }

        for (int i = 0; i < 784; ++i) {
            images[iex*784 + i] = -1.0f + image[i] * (2.0f/255);
        }

        for (int i = 0; i < 10; ++i) {
            labels[iex*10 + i] = i == label ? 1.0f : 0.0f;
        }
    }

    return true;
}

bool mnist_model_train(const float * images, const float * labels, const int nex, mnist_model & model) {
    const mnist_hparams & hparams = model.hparams;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> nd1{0.0f, 1.0f/sqrtf(hparams.n_input*hparams.n_hidden)};
    std::normal_distribution<float> nd2{0.0f, 1.0f/sqrtf(hparams.n_hidden*hparams.n_classes)};

    struct ggml_init_params params = {
        /*.mem_size   =*/ 100 * 1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    model.ctx = ggml_init(params);
    model.fc1_weight = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, hparams.n_input,  hparams.n_hidden);
    model.fc1_bias   = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32,                   hparams.n_hidden);
    model.fc2_weight = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, hparams.n_hidden, hparams.n_classes);
    model.fc2_bias   = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32,                   hparams.n_classes);

    ggml_set_param(model.ctx, model.fc1_weight);
    ggml_set_param(model.ctx, model.fc1_bias);
    ggml_set_param(model.ctx, model.fc2_weight);
    ggml_set_param(model.ctx, model.fc2_bias);

    ggml_set_name(model.fc1_weight, "fc1_weight");
    ggml_set_name(model.fc1_bias,   "fc1_bias");
    ggml_set_name(model.fc2_weight, "fc2_weight");
    ggml_set_name(model.fc2_bias,   "fc2_bias");

    for (ggml_tensor * t : {model.fc1_weight, model.fc1_bias}) {
        float * data = (float *) t->data;
        const int64_t ne = ggml_nelements(t);

        for (int64_t i = 0; i < ne; ++i) {
            data[i] = nd1(gen);
        }
    }
    for (ggml_tensor * t : {model.fc2_weight, model.fc2_bias}) {
        float * data = (float *) t->data;
        const int64_t ne = ggml_nelements(t);

        for (int64_t i = 0; i < ne; ++i) {
            data[i] = nd2(gen);
        }
    }

    const int nbatch = 100;
    GGML_ASSERT(nex % nbatch == 0);

    struct ggml_tensor * images_batch = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, hparams.n_input, nbatch);
    ggml_set_name(images_batch, "images_batch");
    ggml_set_input(images_batch);

    struct ggml_tensor * labels_batch = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, hparams.n_classes, nbatch);
    ggml_set_name(labels_batch, "labels_batch");
    ggml_set_input(labels_batch);

    // fc1 MLP = Ax + b
    struct ggml_tensor * fc1 = ggml_add(
        model.ctx, ggml_mul_mat(model.ctx, model.fc1_weight, images_batch),
        // model.fc1_bias);
        ggml_repeat(model.ctx, model.fc1_bias, ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, hparams.n_hidden,  nbatch)));
    struct ggml_tensor * fc2 = ggml_add(
        model.ctx, ggml_mul_mat(model.ctx, model.fc2_weight, ggml_relu(model.ctx, fc1)),
        // model.fc2_bias);
        ggml_repeat(model.ctx, model.fc2_bias, ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, hparams.n_classes, nbatch)));

    // // soft max
    // struct ggml_tensor * probs = ggml_soft_max(model.ctx, fc2);
    // ggml_set_name(probs, "probs");
    // GGML_ASSERT(probs->ne[0] == hparams.n_classes);
    // GGML_ASSERT(probs->ne[1] == nbatch);
    // GGML_ASSERT(probs->ne[2] == 1);
    // GGML_ASSERT(probs->ne[3] == 1);

    struct ggml_tensor * loss = ggml_cross_entropy_loss(model.ctx, fc2, labels_batch);
    ggml_set_output(loss);
    GGML_ASSERT(loss->ne[0] == 1);
    GGML_ASSERT(loss->ne[1] == 1);
    GGML_ASSERT(loss->ne[2] == 1);
    GGML_ASSERT(loss->ne[3] == 1);

    struct ggml_cgraph * gf = ggml_new_graph_custom(model.ctx, 16384, true);
    ggml_build_forward_expand(gf, loss);

    struct ggml_cgraph * gb = ggml_graph_dup(model.ctx, gf);
    ggml_build_backward_expand(model.ctx, gf, gb, true);

    struct ggml_opt_context opt_ctx;
    struct ggml_opt_params  opt_pars = ggml_opt_default_params(GGML_OPT_TYPE_ADAM);
    opt_pars.print_forward_graph = false;
    opt_pars.print_backward_graph = false;
    opt_pars.n_threads = 16;
    ggml_opt_init(model.ctx, &opt_ctx, opt_pars, 0);

    for (int epoch = 0; epoch < 3; ++epoch) {
        int ncorrect = 0;
        double loss_sum = 0.0;

        for (int iex0 = 0; iex0 < nex; iex0 += nbatch) {
            memcpy(images_batch->data, images + iex0*hparams.n_input,   ggml_nbytes(images_batch));
            memcpy(labels_batch->data, labels + iex0*hparams.n_classes, ggml_nbytes(labels_batch));

            // ggml_graph_compute_with_ctx(model.ctx, gf, 4);

            enum ggml_opt_result opt_result = ggml_opt_resume_g(model.ctx, &opt_ctx, loss, gf, gb, NULL, NULL);
            // enum ggml_opt_result opt_result = ggml_opt(model.ctx, opt_pars, loss);
            GGML_ASSERT(opt_result == GGML_OPT_RESULT_OK);

            // for (int j = 0; j < hparams.n_classes; ++j) {
            //     fprintf(stderr, "%d: %f <-> %f\n", j, ((float *) fc2->data)[j], ((float *) labels_batch->data)[j]);
            // }

            for (int iexb = 0; iexb < nbatch; ++iexb) {
                const float * ptr_p = (const float *) fc2->data          + iexb*hparams.n_classes;
                const float * ptr_l = (const float *) labels_batch->data + iexb*hparams.n_classes;
                const int prediction = std::max_element(ptr_p, ptr_p + hparams.n_classes) - ptr_p;
                const int label      = std::max_element(ptr_l, ptr_l + hparams.n_classes) - ptr_l;
                if (prediction == label) {
                    ncorrect++;
                }
            }

            loss_sum += *((float *) loss->data);
            fprintf(stderr, "%d ", iex0);
        }
        fprintf(stderr, "\n");

        const double loss_mean = loss_sum / (nex/nbatch);
        const float percent_correct = 100.0f * ncorrect/nex;
        fprintf(stderr, "epoch=%d train_loss=%lf train_acc=%.2f%%\n", epoch, loss_mean, percent_correct);
    }

    return true;
}

// load the model's weights from a file
bool mnist_model_save(const std::string & fname, mnist_model & model) {
    printf("%s: saving model to '%s'\n", __func__, fname.c_str());

    gguf_context * gguf_ctx = gguf_init_empty();
    gguf_add_tensor(gguf_ctx, model.fc1_weight);
    gguf_add_tensor(gguf_ctx, model.fc1_bias);
    gguf_add_tensor(gguf_ctx, model.fc2_weight);
    gguf_add_tensor(gguf_ctx, model.fc2_bias);
    gguf_write_to_file(gguf_ctx, fname.c_str(), false);

    return true;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s models/mnist/train-images-idx3-ubyte models/mnist/train-labels-idx1-ubyte models/mnist/ggml-model-f32.gguf\n", argv[0]);
        exit(0);
    }

    const int nex = 5000;

    const mnist_hparams hparams;
    float * images = (float *) malloc(nex*hparams.n_input  *sizeof(float));
    float * labels = (float *) malloc(nex*hparams.n_classes*sizeof(float));
    mnist_data_load(argv[1], argv[2], images, labels, nex);

    mnist_model model;
    mnist_model_train(images, labels, nex, model);

    if (!mnist_model_save(argv[3], model)) {
        fprintf(stderr, "%s: failed to convert model from '%s' to '%s'\n", __func__, argv[1], argv[2]);
        return 1;
    }

    free(images);
    free(labels);
    return 0;
}
