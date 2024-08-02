#include "ggml.h"

#include "mnist-common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <random>
#include <string>
#include <utility>

bool mnist_image_load(const std::string & fname, float * buf, const int nex, const bool normalize) {
    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "failed to open images file %s\n", fname.c_str());
        return false;
    }
    fin.seekg(16);

    uint8_t image[MNIST_NINPUT];

    for (int iex = 0; iex < nex; ++iex) {
        fin.read((char *) image, sizeof(image));

        for (int i = 0; i < MNIST_NINPUT; ++i) {
            buf[iex*MNIST_NINPUT + i] = image[i];
            if (normalize) {
                buf[iex*MNIST_NINPUT + i] = -1.0f + buf[iex*MNIST_NINPUT + i] * (2.0f/255);
            }
        }
    }

    return true;
}

void mnist_image_print(FILE * stream, const float * image) {
    static_assert(MNIST_NINPUT == 28*28, "Unexpected MNIST_NINPUT");

    for (int row = 0; row < 28; row++) {
        for (int col = 0; col < 28; col++) {
            const int rgb = roundf(image[row*28 + col]);
            fprintf(stream, "\033[48;2;%d;%d;%dm  \033[0m", rgb, rgb, rgb);
        }
        fprintf(stream, "\n");
    }
}

bool mnist_label_load(const std::string & fname, float * buf, const int nex) {
    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "failed to open labels file %s\n", fname.c_str());
        return 0;
    }
    fin.seekg(8);

    uint8_t label;

    for (int iex = 0; iex < nex; ++iex) {
        fin.read((char *) &label, sizeof(label));

        for (int i = 0; i < MNIST_NCLASSES; ++i) {
            buf[iex*MNIST_NCLASSES + i] = i == label ? 1.0f : 0.0f;
        }
    }

    return true;
}

mnist_model mnist_model_init(const std::string & fname, const int nbatch) {
    mnist_model model;
    model.nbatch = nbatch;

    const size_t buf_size = 100 * 1024*1024;
    model.buf_compute = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ model.buf_compute,
        /*.no_alloc   =*/ false,
    };

    model.ctx_compute = ggml_init(params);

    if (fname.empty()) {
        printf("%s: initializing random weights\n", __func__);

        model.fc1_weight = ggml_new_tensor_2d(model.ctx_compute, GGML_TYPE_F32, MNIST_NINPUT,  MNIST_NHIDDEN);
        model.fc1_bias   = ggml_new_tensor_1d(model.ctx_compute, GGML_TYPE_F32,                MNIST_NHIDDEN);
        model.fc2_weight = ggml_new_tensor_2d(model.ctx_compute, GGML_TYPE_F32, MNIST_NHIDDEN, MNIST_NCLASSES);
        model.fc2_bias   = ggml_new_tensor_1d(model.ctx_compute, GGML_TYPE_F32,                MNIST_NCLASSES);
    } else {
        printf("%s: loading model weights from '%s'\n", __func__, fname.c_str());

        struct gguf_init_params params = {
            /*.no_alloc   =*/ false,
            /*.ctx        =*/ &model.ctx_gguf,
        };
        gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);
        if (!ctx) {
            fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
            exit(1);
        }

        model.fc1_weight = ggml_get_tensor(model.ctx_gguf, "fc1.weight");
        GGML_ASSERT(model.fc1_weight->ne[0] == MNIST_NINPUT);
        GGML_ASSERT(model.fc1_weight->ne[1] == MNIST_NHIDDEN);
        GGML_ASSERT(model.fc1_weight->ne[2] == 1);
        GGML_ASSERT(model.fc1_weight->ne[3] == 1);

        model.fc1_bias = ggml_get_tensor(model.ctx_gguf, "fc1.bias");
        GGML_ASSERT(model.fc1_bias->ne[0] == MNIST_NHIDDEN);
        GGML_ASSERT(model.fc1_bias->ne[1] == 1);
        GGML_ASSERT(model.fc1_bias->ne[2] == 1);
        GGML_ASSERT(model.fc1_bias->ne[3] == 1);

        model.fc2_weight = ggml_get_tensor(model.ctx_gguf, "fc2.weight");
        GGML_ASSERT(model.fc2_weight->ne[0] == MNIST_NHIDDEN);
        GGML_ASSERT(model.fc2_weight->ne[1] == MNIST_NCLASSES);
        GGML_ASSERT(model.fc2_weight->ne[2] == 1);
        GGML_ASSERT(model.fc2_weight->ne[3] == 1);

        model.fc2_bias = ggml_get_tensor(model.ctx_gguf, "fc2.bias");
        GGML_ASSERT(model.fc2_bias->ne[0] == MNIST_NCLASSES);
        GGML_ASSERT(model.fc2_bias->ne[1] == 1);
        GGML_ASSERT(model.fc2_bias->ne[2] == 1);
        GGML_ASSERT(model.fc2_bias->ne[3] == 1);
    }

    ggml_set_param(model.ctx_compute, model.fc1_weight);
    ggml_set_param(model.ctx_compute, model.fc1_bias);
    ggml_set_param(model.ctx_compute, model.fc2_weight);
    ggml_set_param(model.ctx_compute, model.fc2_bias);

    ggml_set_name(model.fc1_weight, "fc1.weight");
    ggml_set_name(model.fc1_bias,   "fc1.bias");
    ggml_set_name(model.fc2_weight, "fc2.weight");
    ggml_set_name(model.fc2_bias,   "fc2.bias");

    model.images = ggml_new_tensor_2d(model.ctx_compute, GGML_TYPE_F32, MNIST_NINPUT, model.nbatch);
    ggml_set_input(model.images);
    ggml_set_name(model.images, "images");

    model.labels = ggml_new_tensor_2d(model.ctx_compute, GGML_TYPE_F32, MNIST_NCLASSES, model.nbatch);
    ggml_set_input(model.labels);
    ggml_set_name(model.labels, "labels");

    ggml_tensor * fc1_bias = model.fc1_bias;
    if (model.nbatch > 1) {
        fc1_bias = ggml_repeat(model.ctx_compute,
            model.fc1_bias,
            ggml_new_tensor_2d(model.ctx_compute, GGML_TYPE_F32, MNIST_NHIDDEN, model.nbatch));
    }
    ggml_tensor * fc2_bias = model.fc2_bias;
    if (model.nbatch > 1) {
        fc2_bias = ggml_repeat(model.ctx_compute,
            model.fc2_bias,
            ggml_new_tensor_2d(model.ctx_compute, GGML_TYPE_F32, MNIST_NCLASSES, model.nbatch));
    }

    ggml_tensor * fc1 = ggml_relu(model.ctx_compute, ggml_add(model.ctx_compute,
        ggml_mul_mat(model.ctx_compute, model.fc1_weight, model.images),
        fc1_bias));
    model.fc2 = ggml_add(model.ctx_compute,
        ggml_mul_mat(model.ctx_compute, model.fc2_weight, fc1),
        fc2_bias);

    model.probs = ggml_soft_max(model.ctx_compute, model.fc2);
    ggml_set_output(model.probs);
    ggml_set_name(model.probs, "probs");

    model.loss = ggml_cross_entropy_loss(model.ctx_compute, model.fc2, model.labels);
    ggml_set_output(model.loss);
    ggml_set_name(model.loss, "loss");

    return model;
}

void mnist_model_free(mnist_model & model) {
    ggml_free(model.ctx_compute);
    free(model.buf_compute);
    ggml_free(model.ctx_gguf);
}

mnist_eval_result mnist_model_eval(const mnist_model & model, const float * images, const float * labels, const int nex) {
    mnist_eval_result result;

    struct ggml_cgraph * gf = ggml_new_graph(model.ctx_compute);
    ggml_build_forward_expand(gf, model.probs);

    for (int iex; iex < nex; ++iex) {
        memcpy(model.images->data, images + iex*MNIST_NINPUT, ggml_nbytes(model.images));
        ggml_graph_compute_with_ctx(model.ctx_compute, gf, 16);

        const float * probs_data = ggml_get_data_f32(model.probs);
        result.pred.push_back(std::max_element(probs_data, probs_data + MNIST_NCLASSES) - probs_data);
    }

    return result;
}

void mnist_model_train(const float * images, const float * labels, const int nex, mnist_model & model) {
    const int64_t t_start_us = ggml_time_us();

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> nd1{0.0f, 1.0f/sqrtf(MNIST_NINPUT*MNIST_NHIDDEN)};
    std::normal_distribution<float> nd2{0.0f, 1.0f/sqrtf(MNIST_NHIDDEN*MNIST_NCLASSES)};

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

    struct ggml_cgraph * gf = ggml_new_graph_custom(model.ctx_compute, 16384, true);
    ggml_build_forward_expand(gf, model.loss);

    struct ggml_cgraph * gb = ggml_graph_dup(model.ctx_compute, gf);
    ggml_build_backward_expand(model.ctx_compute, gf, gb, true);

    struct ggml_opt_context opt_ctx;
    struct ggml_opt_params  opt_pars = ggml_opt_default_params(GGML_OPT_TYPE_ADAM);
    opt_pars.print_forward_graph = false;
    opt_pars.print_backward_graph = false;
    opt_pars.n_threads = 16;
    opt_pars.adam.n_iter = 1;
    ggml_opt_init(model.ctx_compute, &opt_ctx, opt_pars, 0);

    for (int epoch = 0; epoch < 20; ++epoch) {
        fprintf(stderr, "%s: epoch %d start...", __func__, epoch, nex);
        const int64_t t_start_us = ggml_time_us();
        mnist_eval_result result;
        double loss_sum = 0.0;

        for (int iex0 = 0; iex0 < nex; iex0 += model.nbatch) {
            memcpy(model.images->data,  images + iex0*MNIST_NINPUT,   ggml_nbytes(model.images));
            memcpy(model.labels->data, labels + iex0*MNIST_NCLASSES, ggml_nbytes(model.labels));

            enum ggml_opt_result opt_result = ggml_opt_resume_g(model.ctx_compute, &opt_ctx, model.loss, gf, gb, NULL, NULL);
            GGML_ASSERT(opt_result == GGML_OPT_RESULT_OK || opt_result == GGML_OPT_RESULT_DID_NOT_CONVERGE);

            for (int iexb = 0; iexb < model.nbatch; ++iexb) {
                const float * ptr_p = (const float *) model.fc2->data + iexb*MNIST_NCLASSES;
                result.pred.push_back(std::max_element(ptr_p, ptr_p + MNIST_NCLASSES) - ptr_p);
            }

            loss_sum += *((float *) model.loss->data);
        }

        const double loss_mean = loss_sum / (nex/model.nbatch);

        const double percent_correct = 100.0 * mnist_accuracy(result, labels).first;

        const int64_t t_epoch_us = ggml_time_us() - t_start_us;
        const double t_epoch_s = 1e-6*t_epoch_us;
        fprintf(stderr, "done, took %.2lfs, train_loss=%lf, train_acc=%.2f%%\n", epoch, t_epoch_s, loss_mean, percent_correct);
    }

    const int64_t t_total_us = ggml_time_us() - t_start_us;
    const double t_total_s = 1e-6*t_total_us;
    fprintf(stderr, "%s: training took %.2lfs\n", __func__, t_total_s);
}

std::pair<double, double> mnist_accuracy(const mnist_eval_result & result, const float * labels) {
    const size_t nex = result.pred.size();

    size_t ncorrect = 0;
    for (size_t iex = 0; iex < nex; ++iex) {
        const float * labels_iex = labels + iex*MNIST_NCLASSES;
        const int32_t label = std::max_element(labels_iex, labels_iex + MNIST_NCLASSES) - labels_iex;

        ncorrect += result.pred[iex] == label;
    }

    const double fraction_correct = ((double) ncorrect) / ((double) result.pred.size());
    const double uncertainty = sqrt(fraction_correct * (1.0 - fraction_correct) / nex);

    return std::make_pair(fraction_correct, uncertainty);
}

void mnist_model_save(const std::string & fname, mnist_model & model) {
    printf("%s: saving model to '%s'\n", __func__, fname.c_str());

    gguf_context * gguf_ctx = gguf_init_empty();
    gguf_add_tensor(gguf_ctx, model.fc1_weight);
    gguf_add_tensor(gguf_ctx, model.fc1_bias);
    gguf_add_tensor(gguf_ctx, model.fc2_weight);
    gguf_add_tensor(gguf_ctx, model.fc2_bias);
    gguf_write_to_file(gguf_ctx, fname.c_str(), false);
}
