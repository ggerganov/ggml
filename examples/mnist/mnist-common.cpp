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
            buf[iex*MNIST_NINPUT + i] = image[i] / 255.0f; // Normalize to [0, 1]
        }
    }

    return true;
}

void mnist_image_print(FILE * stream, const float * image) {
    static_assert(MNIST_NINPUT == 28*28, "Unexpected MNIST_NINPUT");

    for (int row = 0; row < 28; row++) {
        for (int col = 0; col < 28; col++) {
            const int rgb = roundf(255.0f * image[row*28 + col]);
#ifdef _WIN32
            fprintf(stream, "%s", rgb >= 220 ? "##" : "__");                // Represented via text.
#else
            fprintf(stream, "\033[48;2;%d;%d;%dm  \033[0m", rgb, rgb, rgb); // Represented via colored blocks.
#endif // _WIN32
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

mnist_eval_result mnist_graph_eval(const std::string & fname, const float * images, const float * labels, const int nex, const int nthreads) {
    fprintf(stderr, "%s: trying to load a ggml graph from %s\n", __func__, fname.c_str());
    mnist_eval_result result;

    struct ggml_context * ctx_data = nullptr;
    struct ggml_context * ctx_eval = nullptr;

    struct ggml_cgraph * gf;
    {
        const int64_t t_start_us = ggml_time_us();

        gf = ggml_graph_import(fname.c_str(), &ctx_data, &ctx_eval);

        const int64_t t_total_us = ggml_time_us() - t_start_us;
        const double t_total_ms = 1e-3*t_total_us;
        if (gf) {
            fprintf(stderr, "%s: graph import took %.2lf ms\n", __func__, t_total_ms);
        }
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
    GGML_ASSERT(images_batch->ne[0] == MNIST_NINPUT || (images_batch->ne[0] == MNIST_HW && images_batch->ne[1] == MNIST_HW));

    struct ggml_tensor * logits_batch = ggml_graph_get_tensor(gf, "logits");
    GGML_ASSERT(logits_batch);
    GGML_ASSERT(logits_batch->ne[0] == MNIST_NCLASSES);
    GGML_ASSERT(logits_batch->ne[2] == 1);
    GGML_ASSERT(logits_batch->ne[3] == 1);

    GGML_ASSERT(images_batch->ne[1] == logits_batch->ne[1] || images_batch->ne[3] == logits_batch->ne[1]);
    const int nbatch = logits_batch->ne[1];
    GGML_ASSERT(nex % nbatch == 0);

    struct ggml_tensor * loss = ggml_graph_get_tensor(gf, "loss");

    {
        const int64_t t_start_us = ggml_time_us();

        for (int iex0; iex0 < nex; iex0 += nbatch) {
            memcpy(images_batch->data, images + iex0*MNIST_NINPUT, ggml_nbytes(images_batch));
            ggml_graph_compute_with_ctx(ctx_compute, gf, nthreads);

            for (int iexb = 0; iexb < nbatch; ++iexb) {
                const float * probs_data = ggml_get_data_f32(logits_batch) + iexb*MNIST_NCLASSES;

                result.pred.push_back(std::max_element(probs_data, probs_data + MNIST_NCLASSES) - probs_data);
            }

            result.loss.push_back(*ggml_get_data_f32(loss));
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

mnist_model mnist_model_init_from_file(const std::string & fname) {
    mnist_model model;
    fprintf(stderr, "%s: loading model weights from '%s'\n", __func__, fname.c_str());

    struct gguf_init_params params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &model.ctx_weight,
    };
    gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);
    if (!ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        exit(1);
    }
    model.arch = gguf_get_val_str(ctx, gguf_find_key(ctx, "general.architecture"));
    fprintf(stderr, "%s: model arch is %s\n", __func__, model.arch.c_str());

    if (model.arch == "mnist-fc") {
        model.fc1_weight = ggml_get_tensor(model.ctx_weight, "fc1.weight");
        GGML_ASSERT(model.fc1_weight->ne[0] == MNIST_NINPUT);
        GGML_ASSERT(model.fc1_weight->ne[1] == MNIST_NHIDDEN);
        GGML_ASSERT(model.fc1_weight->ne[2] == 1);
        GGML_ASSERT(model.fc1_weight->ne[3] == 1);

        model.fc1_bias = ggml_get_tensor(model.ctx_weight, "fc1.bias");
        GGML_ASSERT(model.fc1_bias->ne[0] == MNIST_NHIDDEN);
        GGML_ASSERT(model.fc1_bias->ne[1] == 1);
        GGML_ASSERT(model.fc1_bias->ne[2] == 1);
        GGML_ASSERT(model.fc1_bias->ne[3] == 1);

        model.fc2_weight = ggml_get_tensor(model.ctx_weight, "fc2.weight");
        GGML_ASSERT(model.fc2_weight->ne[0] == MNIST_NHIDDEN);
        GGML_ASSERT(model.fc2_weight->ne[1] == MNIST_NCLASSES);
        GGML_ASSERT(model.fc2_weight->ne[2] == 1);
        GGML_ASSERT(model.fc2_weight->ne[3] == 1);

        model.fc2_bias = ggml_get_tensor(model.ctx_weight, "fc2.bias");
        GGML_ASSERT(model.fc2_bias->ne[0] == MNIST_NCLASSES);
        GGML_ASSERT(model.fc2_bias->ne[1] == 1);
        GGML_ASSERT(model.fc2_bias->ne[2] == 1);
        GGML_ASSERT(model.fc2_bias->ne[3] == 1);
    } else if (model.arch == "mnist-cnn") {
        model.conv1_kernel = ggml_get_tensor(model.ctx_weight, "conv1.kernel");
        GGML_ASSERT(model.conv1_kernel->type == GGML_TYPE_F32);
        GGML_ASSERT(model.conv1_kernel->ne[0] == 3);
        GGML_ASSERT(model.conv1_kernel->ne[1] == 3);
        GGML_ASSERT(model.conv1_kernel->ne[2] == 1);
        GGML_ASSERT(model.conv1_kernel->ne[3] == MNIST_CNN_NCB);

        model.conv1_bias = ggml_get_tensor(model.ctx_weight, "conv1.bias");
        GGML_ASSERT(model.conv1_bias->type == GGML_TYPE_F32);
        GGML_ASSERT(model.conv1_bias->ne[0] == MNIST_HW);
        GGML_ASSERT(model.conv1_bias->ne[1] == MNIST_HW);
        GGML_ASSERT(model.conv1_bias->ne[2] == MNIST_CNN_NCB);
        GGML_ASSERT(model.conv1_bias->ne[3] == 1);

        model.conv2_kernel = ggml_get_tensor(model.ctx_weight, "conv2.kernel");
        GGML_ASSERT(model.conv2_kernel->type == GGML_TYPE_F32);
        GGML_ASSERT(model.conv2_kernel->ne[0] == 3);
        GGML_ASSERT(model.conv2_kernel->ne[1] == 3);
        GGML_ASSERT(model.conv2_kernel->ne[2] == MNIST_CNN_NCB);
        GGML_ASSERT(model.conv2_kernel->ne[3] == MNIST_CNN_NCB*2);

        model.conv2_bias = ggml_get_tensor(model.ctx_weight, "conv2.bias");
        GGML_ASSERT(model.conv2_bias->type == GGML_TYPE_F32);
        GGML_ASSERT(model.conv2_bias->ne[0] == MNIST_HW/2);
        GGML_ASSERT(model.conv2_bias->ne[1] == MNIST_HW/2);
        GGML_ASSERT(model.conv2_bias->ne[2] == MNIST_CNN_NCB*2);
        GGML_ASSERT(model.conv2_bias->ne[3] == 1);

        model.dense_weight = ggml_get_tensor(model.ctx_weight, "dense.weight");
        GGML_ASSERT(model.dense_weight->type == GGML_TYPE_F32);
        GGML_ASSERT(model.dense_weight->ne[0] == (MNIST_HW/4)*(MNIST_HW/4)*(MNIST_CNN_NCB*2));
        GGML_ASSERT(model.dense_weight->ne[1] == MNIST_NCLASSES);
        GGML_ASSERT(model.dense_weight->ne[2] == 1);
        GGML_ASSERT(model.dense_weight->ne[3] == 1);

        model.dense_bias = ggml_get_tensor(model.ctx_weight, "dense.bias");
        GGML_ASSERT(model.dense_bias->type == GGML_TYPE_F32);
        GGML_ASSERT(model.dense_bias->ne[0] == MNIST_NCLASSES);
        GGML_ASSERT(model.dense_bias->ne[1] == 1);
        GGML_ASSERT(model.dense_bias->ne[2] == 1);
        GGML_ASSERT(model.dense_bias->ne[3] == 1);
    } else {
        fprintf(stderr, "%s: unknown model arch: %s\n", __func__, model.arch.c_str());
    }
    fprintf(stderr, "%s: successfully loaded weights from %s\n", __func__, fname.c_str());
    return model;
}

mnist_model mnist_model_init_random(const std::string & arch) {
    mnist_model model;
    model.arch = arch;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> nd{0.0f, 1e-2f};
    std::vector<ggml_tensor *> init_tensors;

    if (model.arch == "mnist-fc") {
        fprintf(stderr, "%s: initializing random weights for a fully connected model\n", __func__);

        model.fc1_weight = ggml_new_tensor_2d(model.ctx_weight, GGML_TYPE_F32, MNIST_NINPUT,  MNIST_NHIDDEN);
        model.fc1_bias   = ggml_new_tensor_1d(model.ctx_weight, GGML_TYPE_F32,                MNIST_NHIDDEN);
        model.fc2_weight = ggml_new_tensor_2d(model.ctx_weight, GGML_TYPE_F32, MNIST_NHIDDEN, MNIST_NCLASSES);
        model.fc2_bias   = ggml_new_tensor_1d(model.ctx_weight, GGML_TYPE_F32,                MNIST_NCLASSES);

        ggml_set_name(model.fc1_weight, "fc1.weight");
        ggml_set_name(model.fc1_bias,   "fc1.bias");
        ggml_set_name(model.fc2_weight, "fc2.weight");
        ggml_set_name(model.fc2_bias,   "fc2.bias");

        init_tensors.push_back(model.fc1_weight);
        init_tensors.push_back(model.fc1_bias);
        init_tensors.push_back(model.fc2_weight);
        init_tensors.push_back(model.fc2_bias);
    } else if (model.arch == "mnist-cnn") {
        model.conv1_kernel = ggml_new_tensor_4d(model.ctx_weight, GGML_TYPE_F32, 3, 3, 1, MNIST_CNN_NCB);
        model.conv1_bias   = ggml_new_tensor_3d(model.ctx_weight, GGML_TYPE_F32, 1, 1,    MNIST_CNN_NCB);
        model.conv2_kernel = ggml_new_tensor_4d(model.ctx_weight, GGML_TYPE_F32, 3, 3, MNIST_CNN_NCB, MNIST_CNN_NCB*2);
        model.conv2_bias   = ggml_new_tensor_3d(model.ctx_weight, GGML_TYPE_F32, 1, 1,                MNIST_CNN_NCB*2);
        model.dense_weight = ggml_new_tensor_2d(model.ctx_weight, GGML_TYPE_F32, (MNIST_HW/4)*(MNIST_HW/4)*(MNIST_CNN_NCB*2), MNIST_NCLASSES);
        model.dense_bias   = ggml_new_tensor_1d(model.ctx_weight, GGML_TYPE_F32, MNIST_NCLASSES);

        ggml_set_name(model.conv1_kernel, "conv1.kernel");
        ggml_set_name(model.conv1_bias,   "conv1.bias");
        ggml_set_name(model.conv2_kernel, "conv2.kernel");
        ggml_set_name(model.conv2_bias,   "conv2.bias");
        ggml_set_name(model.dense_weight, "dense.weight");
        ggml_set_name(model.dense_bias,   "dense.bias");

        init_tensors.push_back(model.conv1_kernel);
        init_tensors.push_back(model.conv1_bias);
        init_tensors.push_back(model.conv2_kernel);
        init_tensors.push_back(model.conv2_bias);
        init_tensors.push_back(model.dense_weight);
        init_tensors.push_back(model.dense_bias);
    } else {
        fprintf(stderr, "%s: unknown model arch: %s\n", __func__, model.arch.c_str());
    }

    for (ggml_tensor * t : init_tensors) {
        GGML_ASSERT(t->type == GGML_TYPE_F32);
        float * data = ggml_get_data_f32(t);
        const int64_t ne = ggml_nelements(t);

        for (int64_t i = 0; i < ne; ++i) {
            data[i] = nd(gen);
        }
    }

    return model;
}

void mnist_model_build(mnist_model & model, const int nbatch) {
    model.nbatch = nbatch;

    if (model.arch == "mnist-fc") {
        ggml_set_param(model.ctx_compute, model.fc1_weight);
        ggml_set_param(model.ctx_compute, model.fc1_bias);
        ggml_set_param(model.ctx_compute, model.fc2_weight);
        ggml_set_param(model.ctx_compute, model.fc2_bias);

        model.images = ggml_new_tensor_2d(model.ctx_compute, GGML_TYPE_F32, MNIST_NINPUT, model.nbatch);
        ggml_set_input(model.images);
        ggml_set_name(model.images, "images");

        ggml_tensor * fc1 = ggml_relu(model.ctx_compute, ggml_add(model.ctx_compute,
            ggml_mul_mat(model.ctx_compute, model.fc1_weight, model.images),
            model.fc1_bias));
        model.logits = ggml_add(model.ctx_compute,
            ggml_mul_mat(model.ctx_compute, model.fc2_weight, fc1),
            model.fc2_bias);
    } else if (model.arch == "mnist-cnn") {
        ggml_set_param(model.ctx_compute, model.conv1_kernel);
        ggml_set_param(model.ctx_compute, model.conv1_bias);
        ggml_set_param(model.ctx_compute, model.conv2_kernel);
        ggml_set_param(model.ctx_compute, model.conv2_bias);
        ggml_set_param(model.ctx_compute, model.dense_weight);
        ggml_set_param(model.ctx_compute, model.dense_bias);

        model.images = ggml_new_tensor_4d(model.ctx_compute, GGML_TYPE_F32, 28, 28, 1, model.nbatch);
        ggml_set_input(model.images);
        ggml_set_name(model.images, "images");

        struct ggml_tensor * conv1_out = ggml_relu(model.ctx_compute, ggml_add(model.ctx_compute,
            ggml_conv_2d(model.ctx_compute, model.conv1_kernel, model.images, 1, 1, 1, 1, 1, 1),
            model.conv1_bias));
        GGML_ASSERT(conv1_out->ne[0] == MNIST_HW);
        GGML_ASSERT(conv1_out->ne[1] == MNIST_HW);
        GGML_ASSERT(conv1_out->ne[2] == MNIST_CNN_NCB);
        GGML_ASSERT(conv1_out->ne[3] == model.nbatch);

        struct ggml_tensor * conv2_in = ggml_pool_2d(model.ctx_compute, conv1_out, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
        GGML_ASSERT(conv2_in->ne[0] == MNIST_HW/2);
        GGML_ASSERT(conv2_in->ne[1] == MNIST_HW/2);
        GGML_ASSERT(conv2_in->ne[2] == MNIST_CNN_NCB);
        GGML_ASSERT(conv2_in->ne[3] == model.nbatch);

        struct ggml_tensor * conv2_out = ggml_relu(model.ctx_compute, ggml_add(model.ctx_compute,
            ggml_conv_2d(model.ctx_compute, model.conv2_kernel, conv2_in, 1, 1, 1, 1, 1, 1),
            model.conv2_bias));
        GGML_ASSERT(conv2_out->ne[0] == MNIST_HW/2);
        GGML_ASSERT(conv2_out->ne[1] == MNIST_HW/2);
        GGML_ASSERT(conv2_out->ne[2] == MNIST_CNN_NCB*2);
        GGML_ASSERT(conv2_out->ne[3] == model.nbatch);

        struct ggml_tensor * dense_in = ggml_pool_2d(model.ctx_compute, conv2_out, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
        GGML_ASSERT(dense_in->ne[0] == MNIST_HW/4);
        GGML_ASSERT(dense_in->ne[1] == MNIST_HW/4);
        GGML_ASSERT(dense_in->ne[2] == MNIST_CNN_NCB*2);
        GGML_ASSERT(dense_in->ne[3] == model.nbatch);

        dense_in = ggml_reshape_2d(model.ctx_compute,
            ggml_cont(model.ctx_compute, ggml_permute(model.ctx_compute, dense_in, 1, 2, 0, 3)),
            (MNIST_HW/4)*(MNIST_HW/4)*(MNIST_CNN_NCB*2), model.nbatch);
        GGML_ASSERT(dense_in->ne[0] == (MNIST_HW/4)*(MNIST_HW/4)*(MNIST_CNN_NCB*2));
        GGML_ASSERT(dense_in->ne[1] == model.nbatch);
        GGML_ASSERT(dense_in->ne[2] == 1);
        GGML_ASSERT(dense_in->ne[3] == 1);

        model.logits = ggml_add(model.ctx_compute, ggml_mul_mat(model.ctx_compute, model.dense_weight, dense_in), model.dense_bias);
    } else {
        GGML_ASSERT(false);
    }

    ggml_set_output(model.logits);
    ggml_set_name(model.logits, "logits");
    GGML_ASSERT(model.logits->type == GGML_TYPE_F32);
    GGML_ASSERT(model.logits->ne[0] == MNIST_NCLASSES);
    GGML_ASSERT(model.logits->ne[1] == model.nbatch);
    GGML_ASSERT(model.logits->ne[2] == 1);
    GGML_ASSERT(model.logits->ne[3] == 1);

    model.probs = ggml_soft_max(model.ctx_compute, model.logits);
    ggml_set_output(model.probs);
    ggml_set_name(model.probs, "probs");
    GGML_ASSERT(model.probs->type == GGML_TYPE_F32);
    GGML_ASSERT(model.probs->ne[0] == MNIST_NCLASSES);
    GGML_ASSERT(model.probs->ne[1] == model.nbatch);
    GGML_ASSERT(model.probs->ne[2] == 1);
    GGML_ASSERT(model.probs->ne[3] == 1);

    model.labels = ggml_new_tensor_2d(model.ctx_compute, GGML_TYPE_F32, MNIST_NCLASSES, model.nbatch);
    ggml_set_input(model.labels);
    ggml_set_name(model.labels, "labels");

    model.loss = ggml_cross_entropy_loss(model.ctx_compute, model.logits, model.labels);
    ggml_set_output(model.loss);
    ggml_set_name(model.loss, "loss");
    GGML_ASSERT(model.loss->type == GGML_TYPE_F32);
    GGML_ASSERT(model.loss->ne[0] == 1);
    GGML_ASSERT(model.loss->ne[1] == 1);
    GGML_ASSERT(model.loss->ne[2] == 1);
    GGML_ASSERT(model.loss->ne[3] == 1);
}

mnist_eval_result mnist_model_eval(const mnist_model & model, const float * images, const float * labels, const int nex, const int nthreads) {
    mnist_eval_result result;

    struct ggml_cgraph * gf = ggml_new_graph(model.ctx_compute);
    ggml_build_forward_expand(gf, model.loss);

    {
        const int64_t t_start_us = ggml_time_us();

        GGML_ASSERT(nex % model.nbatch == 0);
        for (int iex0 = 0; iex0 < nex; iex0 += model.nbatch) {
            memcpy(model.images->data, images + iex0*MNIST_NINPUT,   ggml_nbytes(model.images));
            memcpy(model.labels->data, labels + iex0*MNIST_NCLASSES, ggml_nbytes(model.labels));
            ggml_graph_compute_with_ctx(model.ctx_compute, gf, nthreads);

            result.loss.push_back(*ggml_get_data_f32(model.loss));

            for (int iexb = 0; iexb < model.nbatch; ++iexb) {
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

void mnist_model_train(mnist_model & model, const float * images, const float * labels, const int nex, const int nthreads) {
    const int64_t t_start_us = ggml_time_us();

    struct ggml_cgraph * gf = ggml_new_graph_custom(model.ctx_compute, 16384, true);
    ggml_build_forward_expand(gf, model.loss);

    struct ggml_cgraph * gb = ggml_graph_dup(model.ctx_compute, gf);
    ggml_build_backward_expand(model.ctx_compute, gf, gb, true);

    struct ggml_opt_context opt_ctx;
    struct ggml_opt_params  opt_pars = ggml_opt_default_params(GGML_OPT_TYPE_ADAM);
    opt_pars.print_forward_graph = false;
    opt_pars.print_backward_graph = false;
    opt_pars.n_threads = nthreads;
    opt_pars.adam.n_iter = 1; // per call of ggml_opt_resume_g
    ggml_opt_init(model.ctx_compute, &opt_ctx, opt_pars, 0);

    for (int epoch = 0; epoch < 20; ++epoch) {
        fprintf(stderr, "%s: epoch %d start...", __func__, epoch);
        const int64_t t_start_us = ggml_time_us();
        mnist_eval_result result;
        for (int iex0 = 0; iex0 < nex; iex0 += model.nbatch) {
            memcpy(model.images->data, images + iex0*MNIST_NINPUT,   ggml_nbytes(model.images));
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

    std::string fname = model.arch + "-f32.ggml";
    fprintf(stderr, "%s: saving the ggml graph for the forward pass to %s\n", __func__, fname.c_str());
    ggml_graph_export(gf, fname.c_str());
}

void mnist_model_save(mnist_model & model, const std::string & fname) {
    printf("%s: saving model to '%s'\n", __func__, fname.c_str());

    gguf_context * gguf_ctx = gguf_init_empty();
    gguf_set_val_str(gguf_ctx, "general.architecture", model.arch.c_str());

    if (model.arch == "mnist-fc") {
        gguf_add_tensor(gguf_ctx, model.fc1_weight);
        gguf_add_tensor(gguf_ctx, model.fc1_bias);
        gguf_add_tensor(gguf_ctx, model.fc2_weight);
        gguf_add_tensor(gguf_ctx, model.fc2_bias);
    } else if (model.arch == "mnist-cnn") {
        gguf_add_tensor(gguf_ctx, model.conv1_kernel);
        gguf_add_tensor(gguf_ctx, model.conv1_bias);
        gguf_add_tensor(gguf_ctx, model.conv2_kernel);
        gguf_add_tensor(gguf_ctx, model.conv2_bias);
        gguf_add_tensor(gguf_ctx, model.dense_weight);
        gguf_add_tensor(gguf_ctx, model.dense_bias);
    } else {
        GGML_ASSERT(false);
    }
    gguf_write_to_file(gguf_ctx, fname.c_str(), false);
}

std::pair<double, double> mnist_loss(const mnist_eval_result & result) {
    const size_t nbatches = result.loss.size();
    GGML_ASSERT(nbatches >= 1);

    double sum         = 0.0;
    double sum_squared = 0.0;

    for (const float & loss : result.loss) {
        sum         += loss;
        sum_squared += loss*loss;
    }

    const double mean        = sum/nbatches;
    const double uncertainty = sqrt((sum_squared/nbatches - mean*mean) / (nbatches - 1));

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
    const double uncertainty      = sqrt(fraction_correct * (1.0 - fraction_correct) / (nex - 1));

    return std::make_pair(fraction_correct, uncertainty);
}

#ifdef __cplusplus
extern "C" {
#endif

int wasm_eval(uint8_t * digitPtr) {
    std::vector<float> digit(digitPtr, digitPtr + MNIST_NINPUT);
    std::vector<float> labels(MNIST_NCLASSES);

    mnist_model model = mnist_model_init_from_file("mnist-f32.gguf");
    mnist_model_build(model, 1);
    mnist_eval_result result = mnist_model_eval(model, digit.data(), labels.data(), 1, 1);

    return result.pred[0];
}

int wasm_random_digit(char * digitPtr) {
    auto fin = std::ifstream("t10k-images-idx3-ubyte", std::ios::binary);
    if (!fin) {
        fprintf(stderr, "failed to open digits file\n");
        return 0;
    }
    srand(time(NULL));

    // Seek to a random digit: 16-byte header + 28*28 * (random 0 - 10000)
    fin.seekg(16 + MNIST_NINPUT * (rand() % MNIST_NTEST));
    fin.read(digitPtr, MNIST_NINPUT);

    return 1;
}

#ifdef __cplusplus
}
#endif
