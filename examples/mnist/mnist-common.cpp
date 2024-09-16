#include "ggml-alloc.h"
#include "ggml-backend.h"
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

    struct ggml_tensor * labels_batch = ggml_graph_get_tensor(gf, "labels");
    GGML_ASSERT(labels_batch);
    GGML_ASSERT(labels_batch->ne[0] == MNIST_NCLASSES);
    GGML_ASSERT(labels_batch->ne[2] == 1);
    GGML_ASSERT(labels_batch->ne[3] == 1);

    const int nbatch = labels_batch->ne[1];
    GGML_ASSERT(nex % nbatch == 0);

    struct ggml_tensor * logits_batch = ggml_graph_get_tensor(gf, "logits");
    GGML_ASSERT(logits_batch);
    GGML_ASSERT(logits_batch->ne[0] == MNIST_NCLASSES);
    GGML_ASSERT(logits_batch->ne[1] == nbatch);
    GGML_ASSERT(logits_batch->ne[2] == 1);
    GGML_ASSERT(logits_batch->ne[3] == 1);

    GGML_ASSERT(images_batch->ne[1] == logits_batch->ne[1] || images_batch->ne[3] == logits_batch->ne[1]);

    struct ggml_tensor * loss = ggml_graph_get_tensor(gf, "loss");

    {
        const int64_t t_start_us = ggml_time_us();

        for (int iex0; iex0 < nex; iex0 += nbatch) {
            memcpy(images_batch->data, images + iex0*MNIST_NINPUT,   ggml_nbytes(images_batch));
            memcpy(labels_batch->data, labels + iex0*MNIST_NCLASSES, ggml_nbytes(labels_batch));
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

// Temporary util function for loading data from GGUF to a backend != CPU until GGML itself provides this functionality:
bool load_from_gguf(const char * fname, struct ggml_context * ctx_ggml, struct gguf_context * ctx_gguf) {
    FILE * f = ggml_fopen(fname, "rb");
    if (!f) {
        return false;
    }

    const size_t buf_size = 4*1024*1024;
    void * buf = malloc(buf_size);

    const int n_tensors = gguf_get_n_tensors(ctx_gguf);
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);

        struct ggml_tensor * tensor = ggml_get_tensor(ctx_ggml, name);
        if (!tensor) {
            continue;
        }

        const size_t offs = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);

        if (fseek(f, offs, SEEK_SET) != 0) {
            fclose(f);
            free(buf);
            return false;
        }

        const size_t nbytes = ggml_nbytes(tensor);
        for (size_t pos = 0; pos < nbytes; pos += buf_size) {
            const size_t nbytes_cpy = buf_size < nbytes - pos ? buf_size : nbytes - pos;

            if (fread(buf, 1, nbytes_cpy, f) != nbytes_cpy) {
                fclose(f);
                free(buf);
                return false;
            }

            ggml_backend_tensor_set(tensor, buf, pos, nbytes_cpy);
        }
    }

    fclose(f);
    free(buf);
    return true;
}

mnist_model mnist_model_init_from_file(const std::string & fname, const std::string & backend) {
    mnist_model model(backend);
    fprintf(stderr, "%s: loading model weights from '%s'\n", __func__, fname.c_str());

    struct gguf_context * ctx;
    {
        struct gguf_init_params params = {
            /*.no_alloc   =*/ true,
            /*.ctx        =*/ &model.ctx_weight,
        };
        ctx = gguf_init_from_file(fname.c_str(), params);
        if (!ctx) {
            fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
            exit(1);
        }
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
        GGML_ASSERT(model.conv1_bias->ne[0] == 1);
        GGML_ASSERT(model.conv1_bias->ne[1] == 1);
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
        GGML_ASSERT(model.conv2_bias->ne[0] == 1);
        GGML_ASSERT(model.conv2_bias->ne[1] == 1);
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
    model.buf_weight = ggml_backend_alloc_ctx_tensors(model.ctx_weight, model.backend);

    if(!load_from_gguf(fname.c_str(), model.ctx_weight, ctx)) {
        fprintf(stderr, "%s: loading weights from %s failed\n", __func__, fname.c_str());
        exit(1);
    }

    fprintf(stderr, "%s: successfully loaded weights from %s\n", __func__, fname.c_str());
    return model;
}

mnist_model mnist_model_init_random(const std::string & arch, const std::string & backend) {
    mnist_model model(backend);
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

    model.buf_weight = ggml_backend_alloc_ctx_tensors(model.ctx_weight, model.backend);

    for (ggml_tensor * t : init_tensors) {
        GGML_ASSERT(t->type == GGML_TYPE_F32);
        const int64_t ne = ggml_nelements(t);
        std::vector<float> tmp(ne);

        for (int64_t i = 0; i < ne; ++i) {
            tmp[i] = nd(gen);
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, ggml_nbytes(t));
    }

    return model;
}

void mnist_model_build(mnist_model & model, const int nbatch_logical, const int nbatch_physical) {
    model.nbatch_logical  = nbatch_logical;
    model.nbatch_physical = nbatch_physical;

    if (model.arch == "mnist-fc") {
        ggml_set_param(model.ctx_compute, model.fc1_weight);
        ggml_set_param(model.ctx_compute, model.fc1_bias);
        ggml_set_param(model.ctx_compute, model.fc2_weight);
        ggml_set_param(model.ctx_compute, model.fc2_bias);

        model.images = ggml_new_tensor_2d(model.ctx_compute, GGML_TYPE_F32, MNIST_NINPUT, model.nbatch_physical);
        ggml_set_name(model.images, "images");
        ggml_set_input(model.images);

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

        model.images = ggml_new_tensor_4d(model.ctx_compute, GGML_TYPE_F32, 28, 28, 1, model.nbatch_physical);
        ggml_set_name(model.images, "images");
        ggml_set_input(model.images);

        struct ggml_tensor * conv1_out = ggml_relu(model.ctx_compute, ggml_add(model.ctx_compute,
            ggml_conv_2d(model.ctx_compute, model.conv1_kernel, model.images, 1, 1, 1, 1, 1, 1),
            model.conv1_bias));
        GGML_ASSERT(conv1_out->ne[0] == MNIST_HW);
        GGML_ASSERT(conv1_out->ne[1] == MNIST_HW);
        GGML_ASSERT(conv1_out->ne[2] == MNIST_CNN_NCB);
        GGML_ASSERT(conv1_out->ne[3] == model.nbatch_physical);

        struct ggml_tensor * conv2_in = ggml_pool_2d(model.ctx_compute, conv1_out, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
        GGML_ASSERT(conv2_in->ne[0] == MNIST_HW/2);
        GGML_ASSERT(conv2_in->ne[1] == MNIST_HW/2);
        GGML_ASSERT(conv2_in->ne[2] == MNIST_CNN_NCB);
        GGML_ASSERT(conv2_in->ne[3] == model.nbatch_physical);

        struct ggml_tensor * conv2_out = ggml_relu(model.ctx_compute, ggml_add(model.ctx_compute,
            ggml_conv_2d(model.ctx_compute, model.conv2_kernel, conv2_in, 1, 1, 1, 1, 1, 1),
            model.conv2_bias));
        GGML_ASSERT(conv2_out->ne[0] == MNIST_HW/2);
        GGML_ASSERT(conv2_out->ne[1] == MNIST_HW/2);
        GGML_ASSERT(conv2_out->ne[2] == MNIST_CNN_NCB*2);
        GGML_ASSERT(conv2_out->ne[3] == model.nbatch_physical);

        struct ggml_tensor * dense_in = ggml_pool_2d(model.ctx_compute, conv2_out, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
        GGML_ASSERT(dense_in->ne[0] == MNIST_HW/4);
        GGML_ASSERT(dense_in->ne[1] == MNIST_HW/4);
        GGML_ASSERT(dense_in->ne[2] == MNIST_CNN_NCB*2);
        GGML_ASSERT(dense_in->ne[3] == model.nbatch_physical);

        dense_in = ggml_reshape_2d(model.ctx_compute,
            ggml_cont(model.ctx_compute, ggml_permute(model.ctx_compute, dense_in, 1, 2, 0, 3)),
            (MNIST_HW/4)*(MNIST_HW/4)*(MNIST_CNN_NCB*2), model.nbatch_physical);
        GGML_ASSERT(dense_in->ne[0] == (MNIST_HW/4)*(MNIST_HW/4)*(MNIST_CNN_NCB*2));
        GGML_ASSERT(dense_in->ne[1] == model.nbatch_physical);
        GGML_ASSERT(dense_in->ne[2] == 1);
        GGML_ASSERT(dense_in->ne[3] == 1);

        model.logits = ggml_add(model.ctx_compute, ggml_mul_mat(model.ctx_compute, model.dense_weight, dense_in), model.dense_bias);
    } else {
        GGML_ASSERT(false);
    }

    ggml_set_name(model.logits, "logits");
    ggml_set_output(model.logits);
    GGML_ASSERT(model.logits->type == GGML_TYPE_F32);
    GGML_ASSERT(model.logits->ne[0] == MNIST_NCLASSES);
    GGML_ASSERT(model.logits->ne[1] == model.nbatch_physical);
    GGML_ASSERT(model.logits->ne[2] == 1);
    GGML_ASSERT(model.logits->ne[3] == 1);

    model.probs = ggml_soft_max(model.ctx_compute, model.logits);
    ggml_set_name(model.probs, "probs");
    ggml_set_output(model.probs);
    GGML_ASSERT(model.probs->type == GGML_TYPE_F32);
    GGML_ASSERT(model.probs->ne[0] == MNIST_NCLASSES);
    GGML_ASSERT(model.probs->ne[1] == model.nbatch_physical);
    GGML_ASSERT(model.probs->ne[2] == 1);
    GGML_ASSERT(model.probs->ne[3] == 1);

    model.labels = ggml_new_tensor_2d(model.ctx_compute, GGML_TYPE_F32, MNIST_NCLASSES, model.nbatch_physical);
    ggml_set_name(model.labels, "labels");
    ggml_set_input(model.labels);

    model.loss = ggml_cross_entropy_loss(model.ctx_compute, model.logits, model.labels);
    ggml_set_name(model.loss, "loss");
    ggml_set_output(model.loss);
    ggml_set_loss(model.loss);
    GGML_ASSERT(model.loss->type == GGML_TYPE_F32);
    GGML_ASSERT(model.loss->ne[0] == 1);
    GGML_ASSERT(model.loss->ne[1] == 1);
    GGML_ASSERT(model.loss->ne[2] == 1);
    GGML_ASSERT(model.loss->ne[3] == 1);
}

mnist_eval_result mnist_model_eval(mnist_model & model, const float * images, const float * labels, const int nex) {
    mnist_eval_result result;

    struct ggml_cgraph * gf = ggml_new_graph(model.ctx_compute);
    ggml_build_forward_expand(gf, model.loss);

    model.buf_compute = ggml_backend_alloc_ctx_tensors(model.ctx_compute, model.backend);

    {
        const int64_t t_start_us = ggml_time_us();

        float loss;
        std::vector<float> logits(model.nbatch_physical*MNIST_NCLASSES);

        GGML_ASSERT(sizeof(loss)  == ggml_nbytes(model.loss));
        GGML_ASSERT(logits.size() == ggml_nelements(model.logits));

        GGML_ASSERT(nex % model.nbatch_physical == 0);
        for (int iex0 = 0; iex0 < nex; iex0 += model.nbatch_physical) {
            ggml_backend_tensor_set(model.images, images + iex0*MNIST_NINPUT,   0, ggml_nbytes(model.images));
            ggml_backend_tensor_set(model.labels, labels + iex0*MNIST_NCLASSES, 0, ggml_nbytes(model.labels));

            ggml_backend_graph_compute(model.backend, gf);

            ggml_backend_tensor_get(model.loss,   &loss,         0, ggml_nbytes(model.loss));
            ggml_backend_tensor_get(model.logits, logits.data(), 0, ggml_nbytes(model.logits));

            result.loss.push_back(loss);

            for (int iexb = 0; iexb < model.nbatch_physical; ++iexb) {
                const float * logits_iexb = logits.data() + iexb*MNIST_NCLASSES;
                result.pred.push_back(std::max_element(logits_iexb, logits_iexb + MNIST_NCLASSES) - logits_iexb);
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

void mnist_model_train(mnist_model & model, const float * images, const float * labels, const int nex, const int nepoch, const float val_split) {
    const int64_t t_start_us = ggml_time_us();

    // gf == graph forward, forward pass only.
    struct ggml_cgraph * gf = ggml_new_graph_custom(model.ctx_compute, GGML_DEFAULT_GRAPH_SIZE, /*grads =*/ true); // Forward pass.
    ggml_build_forward_expand(gf, model.loss);

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    struct ggml_cgraph * gb_grad = ggml_graph_dup(model.ctx_compute, gf);
    ggml_build_backward_expand(model.ctx_compute, gf, gb_grad, /*accumulate =*/ true, false);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    struct ggml_cgraph * gb_opt = ggml_graph_dup(model.ctx_compute, gb_grad);
    ggml_build_opt_adamw(model.ctx_compute, gf, gb_opt, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.0f);

    model.buf_compute = ggml_backend_alloc_ctx_tensors(model.ctx_compute, model.backend);
    ggml_graph_reset(gb_opt); // Set gradients to zero, reset optimizer.

    const int iex_split = ((int)((1.0f - val_split)*nex) / model.nbatch_logical) * model.nbatch_logical;

    for (int epoch = 0; epoch < nepoch; ++epoch) {
        fprintf(stderr, "%s: epoch %02d start...", __func__, epoch);
        const int64_t t_start_us = ggml_time_us();

        float loss;
        std::vector<float> logits(model.nbatch_physical*MNIST_NCLASSES);
        int iex0 = 0;

        mnist_eval_result result_train;
        for (; iex0 < iex_split; iex0 += model.nbatch_physical) {
            ggml_backend_tensor_set(model.images, images + iex0*MNIST_NINPUT,   0, ggml_nbytes(model.images));
            ggml_backend_tensor_set(model.labels, labels + iex0*MNIST_NCLASSES, 0, ggml_nbytes(model.labels));

            // With a period of nbatch_logical/nbatch_physical iterations:
            if ((iex0 + model.nbatch_physical) % model.nbatch_logical != 0) {
                // For the first nbatch_logical/nbatch_physical - 1 iterations, only calculate gradients and accumulate them:
                ggml_backend_graph_compute(model.backend, gb_grad);
            } else {
                // For the last iteration, calculate gradients and also apply the optimizer:
                ggml_backend_graph_compute(model.backend, gb_opt); // gb_opt contains all nodes of gb_grad so no extra call for gb_grad is needed.
                ggml_graph_reset(gb_grad); // Set gradients to zero, do not reset optimizer.
            }

            ggml_backend_tensor_get(model.loss,   &loss,         0, ggml_nbytes(model.loss));
            ggml_backend_tensor_get(model.logits, logits.data(), 0, ggml_nbytes(model.logits));

            result_train.loss.push_back(loss);

            for (int iexb = 0; iexb < model.nbatch_physical; ++iexb) {
                const float * logits_iexb = logits.data() + iexb*MNIST_NCLASSES;
                result_train.pred.push_back(std::max_element(logits_iexb, logits_iexb + MNIST_NCLASSES) - logits_iexb);
            }
        }

        mnist_eval_result result_val;
        for (; iex0 < nex; iex0 += model.nbatch_physical) {
            ggml_backend_tensor_set(model.images, images + iex0*MNIST_NINPUT,   0, ggml_nbytes(model.images));
            ggml_backend_tensor_set(model.labels, labels + iex0*MNIST_NCLASSES, 0, ggml_nbytes(model.labels));

            ggml_backend_graph_compute(model.backend, gf); // For the validation set, only the forward pass is needed.

            ggml_backend_tensor_get(model.loss,   &loss,         0, ggml_nbytes(model.loss));
            ggml_backend_tensor_get(model.logits, logits.data(), 0, ggml_nbytes(model.logits));

            result_val.loss.push_back(loss);

            for (int iexb = 0; iexb < model.nbatch_physical; ++iexb) {
                const float * logits_iexb = logits.data() + iexb*MNIST_NCLASSES;
                result_val.pred.push_back(std::max_element(logits_iexb, logits_iexb + MNIST_NCLASSES) - logits_iexb);
            }
        }

        {
            const double loss_mean = mnist_loss(result_train).first;
            const double percent_correct = 100.0 * mnist_accuracy(result_train, labels + 0*MNIST_NCLASSES).first;

            const int64_t t_epoch_us = ggml_time_us() - t_start_us;
            const double t_epoch_s = 1e-6*t_epoch_us;
            fprintf(stderr, "done, took %.2lfs, train_loss=%.6lf, train_acc=%.2f%%", t_epoch_s, loss_mean, percent_correct);
        }

        if (iex_split < nex) {
            const std::pair<double, double> loss = mnist_loss(result_val);
            const std::pair<double, double> acc  = mnist_accuracy(result_val, labels + iex_split*MNIST_NCLASSES);

            fprintf(stderr, ", val_loss=%.6lf+-%.6lf, val_acc=%.2f+-%.2f%%", loss.first, loss.second, 100.0*acc.first, 100.0*acc.second);
        }
        fprintf(stderr, "\n");
    }

    const int64_t t_total_us = ggml_time_us() - t_start_us;
    const double t_total_s = 1e-6*t_total_us;
    fprintf(stderr, "%s: training took %.2lfs\n", __func__, t_total_s);

    if (ggml_backend_is_cpu(model.backend)) {
        std::string fname = model.arch + "-f32.ggml";
        fprintf(stderr, "%s: saving the GGML graph for the forward pass to %s\n", __func__, fname.c_str());
        ggml_graph_export(gf, fname.c_str());
    } else {
        fprintf(stderr, "%s: not saving the GGML graph for the forward pass because this is only supported for the CPU backend\n", __func__);
    }
}

void mnist_model_save(mnist_model & model, const std::string & fname) {
    printf("%s: saving model to '%s'\n", __func__, fname.c_str());

    struct ggml_context * ggml_ctx;
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ 100 * 1024*1024,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        ggml_ctx = ggml_init(params);
    }

    gguf_context * gguf_ctx = gguf_init_empty();
    gguf_set_val_str(gguf_ctx, "general.architecture", model.arch.c_str());

    std::vector<struct ggml_tensor *> weights;
    if (model.arch == "mnist-fc") {
        weights = {model.fc1_weight, model.fc1_bias, model.fc2_weight, model.fc2_bias};
    } else if (model.arch == "mnist-cnn") {
        weights = {model.conv1_kernel, model.conv1_bias, model.conv2_kernel, model.conv2_bias, model.dense_weight, model.dense_bias};
    } else {
        GGML_ASSERT(false);
    }
    for (struct ggml_tensor * t : weights) {
        struct ggml_tensor * copy = ggml_dup_tensor(ggml_ctx, t);
        ggml_set_name(copy, t->name);
        ggml_backend_tensor_get(t, copy->data, 0, ggml_nbytes(t));
        gguf_add_tensor(gguf_ctx, copy);
    }
    gguf_write_to_file(gguf_ctx, fname.c_str(), false);

    ggml_free(ggml_ctx);
    gguf_free(gguf_ctx);
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

    mnist_model model = mnist_model_init_from_file("mnist-f32.gguf", "CPU");
    mnist_model_build(model, 1, 1);
    mnist_eval_result result = mnist_model_eval(model, digit.data(), labels.data(), 1);

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
