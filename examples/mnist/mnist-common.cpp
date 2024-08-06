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

bool mnist_image_load(const std::string & fname, float * buf, const int nex) {
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

mnist_eval_result mnist_graph_eval(const std::string & fname, const float * images, const float * labels, const int nex) {
    fprintf(stderr, "%s: trying to load a ggml graph from %s\n", __func__, fname.c_str());
    mnist_eval_result result;

    struct ggml_context * ctx_data;
    struct ggml_context * ctx_eval;

    struct ggml_cgraph * gf;
    {
        const int64_t t_start_us = ggml_time_us();

        gf = ggml_graph_import(fname.c_str(), &ctx_data, &ctx_eval);

        const int64_t t_total_us = ggml_time_us() - t_start_us;
        const double t_total_ms = 1e-3*t_total_us;
        fprintf(stderr, "%s: graph import took %.2lf ms\n", __func__, t_total_ms);
    }

    if (!gf) {
        fprintf(stderr, "%s: could not load a ggml graph from %s\n", __func__, fname.c_str());
        return result;
    }
    fprintf(stderr, "%s: successfully loaded a ggml graph from %s\n", __func__, fname.c_str());

    const size_t buf_size = 100 * 1024*1024;
    void * buf_compute = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf_compute,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx_compute = ggml_init(params);

    struct ggml_tensor * images_batch = ggml_graph_get_tensor(gf, "images");
    GGML_ASSERT(images_batch);
    GGML_ASSERT(images_batch->ne[0] == MNIST_NINPUT);
    GGML_ASSERT(images_batch->ne[2] == 1);
    GGML_ASSERT(images_batch->ne[3] == 1);

    struct ggml_tensor * logits_batch = ggml_graph_get_tensor(gf, "logits");
    GGML_ASSERT(logits_batch);
    GGML_ASSERT(logits_batch->ne[0] == MNIST_NCLASSES);
    GGML_ASSERT(logits_batch->ne[2] == 1);
    GGML_ASSERT(logits_batch->ne[3] == 1);

    GGML_ASSERT(images_batch->ne[1] == logits_batch->ne[1]);
    const int nbatch = images_batch->ne[1];
    GGML_ASSERT(nex % nbatch == 0);

    {
        const int64_t t_start_us = ggml_time_us();

        for (int iex0; iex0 < nex; iex0 += nbatch) {
            memcpy(images_batch->data, images + iex0*MNIST_NINPUT, ggml_nbytes(images_batch));
            ggml_graph_compute_with_ctx(ctx_compute, gf, 16);

            for (int iexb = 0; iexb < nbatch; ++iexb) {
                const float * probs_data = ggml_get_data_f32(logits_batch) + iexb*MNIST_NCLASSES;

                result.pred.push_back(std::max_element(probs_data, probs_data + MNIST_NCLASSES) - probs_data);
            }
        }

        const int64_t t_total_us = ggml_time_us() - t_start_us;
        const double t_total_ms = 1e-3*t_total_us;
        fprintf(stderr, "%s: model evaluation on %d images took %.2lf ms, %.2lf us/image\n",
                __func__, nex, t_total_ms, (double) t_total_us/nex);
    }

    ggml_free(ctx_data);
    ggml_free(ctx_eval);
    ggml_free(ctx_compute);
    free(buf_compute);

    result.success = true;
    return result;
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
    model.logits = ggml_add(model.ctx_compute,
        ggml_mul_mat(model.ctx_compute, model.fc2_weight, fc1),
        fc2_bias);
    ggml_set_output(model.logits);
    ggml_set_name(model.logits, "logits");

    model.probs = ggml_soft_max(model.ctx_compute, model.logits);
    ggml_set_output(model.probs);
    ggml_set_name(model.probs, "probs");

    model.loss = ggml_cross_entropy_loss(model.ctx_compute, model.logits, model.labels);
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
    ggml_build_forward_expand(gf, model.loss);

    {
        const int64_t t_start_us = ggml_time_us();

        GGML_ASSERT(nex % MNIST_NBATCH == 0);
        for (int iex0; iex0 < nex; iex0 += MNIST_NBATCH) {
            memcpy(model.images->data, images + iex0*MNIST_NINPUT,   ggml_nbytes(model.images));
            memcpy(model.labels->data, labels + iex0*MNIST_NCLASSES, ggml_nbytes(model.labels));
            ggml_graph_compute_with_ctx(model.ctx_compute, gf, 16);

            result.loss.push_back(*ggml_get_data_f32(model.loss));

            for (int iexb = 0; iexb < MNIST_NBATCH; ++iexb) {
                const float * logits_data = ggml_get_data_f32(model.logits) + iexb*MNIST_NCLASSES;
                result.pred.push_back(std::max_element(logits_data, logits_data + MNIST_NCLASSES) - logits_data);
            }
        }

        const int64_t t_total_us = ggml_time_us() - t_start_us;
        const double t_total_ms = 1e-3*t_total_us;
        fprintf(stderr, "%s: model evaluation on %d images took %.2lf ms, %.2lf us/image\n",
                __func__, nex, t_total_ms, (double) t_total_us/nex);
    }

    result.success = true;
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
        for (int iex0 = 0; iex0 < nex; iex0 += model.nbatch) {
            memcpy(model.images->data,  images + iex0*MNIST_NINPUT,   ggml_nbytes(model.images));
            memcpy(model.labels->data, labels + iex0*MNIST_NCLASSES, ggml_nbytes(model.labels));

            enum ggml_opt_result opt_result = ggml_opt_resume_g(model.ctx_compute, &opt_ctx, model.loss, gf, gb, NULL, NULL);
            GGML_ASSERT(opt_result == GGML_OPT_RESULT_OK || opt_result == GGML_OPT_RESULT_DID_NOT_CONVERGE);

            result.loss.push_back(*ggml_get_data_f32(model.loss));

            for (int iexb = 0; iexb < model.nbatch; ++iexb) {
                const float * ptr_p = (const float *) model.logits->data + iexb*MNIST_NCLASSES;
                result.pred.push_back(std::max_element(ptr_p, ptr_p + MNIST_NCLASSES) - ptr_p);
            }
        }

        const double loss_mean = mnist_loss(result).first;
        const double percent_correct = 100.0 * mnist_accuracy(result, labels).first;

        const int64_t t_epoch_us = ggml_time_us() - t_start_us;
        const double t_epoch_s = 1e-6*t_epoch_us;
        fprintf(stderr, "done, took %.2lfs, train_loss=%.6lf, train_acc=%.2f%%\n", t_epoch_s, loss_mean, percent_correct);
    }

    const int64_t t_total_us = ggml_time_us() - t_start_us;
    const double t_total_s = 1e-6*t_total_us;
    fprintf(stderr, "%s: training took %.2lfs\n", __func__, t_total_s);

    const std::string fname("models/MNIST/mnist-fc-f32.ggml");
    fprintf(stderr, "%s: saving the ggml graph for the forward pass to %s\n", __func__, fname.c_str());
    ggml_graph_export(gf, fname.c_str());
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

std::pair<double, double> mnist_loss(const mnist_eval_result & result) {
    double sum         = 0.0;
    double sum_squared = 0.0;

    for (const float & loss : result.loss) {
        sum         += loss;
        sum_squared += loss*loss;
    }

    const double mean        = sum/result.loss.size();
    const double uncertainty = sqrt((sum_squared/result.loss.size() - mean*mean) / (result.loss.size() - 1));

    return std::make_pair(mean, uncertainty);
}

std::pair<double, double> mnist_accuracy(const mnist_eval_result & result, const float * labels) {
    const size_t nex = result.pred.size();
    GGML_ASSERT(nex >= 1);

    size_t ncorrect = 0;
    for (size_t iex = 0; iex < nex; ++iex) {
        const float * labels_iex = labels + iex*MNIST_NCLASSES;
        const int32_t label = std::max_element(labels_iex, labels_iex + MNIST_NCLASSES) - labels_iex;

        ncorrect += result.pred[iex] == label;
    }

    const double fraction_correct = ((double) ncorrect) / ((double) nex);
    const double uncertainty = sqrt(fraction_correct * (1.0 - fraction_correct) / (nex - 1));

    return std::make_pair(fraction_correct, uncertainty);
}
