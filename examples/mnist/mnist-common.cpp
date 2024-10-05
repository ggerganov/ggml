#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opt.h"

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

bool mnist_image_load(const std::string & fname, ggml_opt_dataset_t dataset) {
    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "failed to open images file %s\n", fname.c_str());
        return false;
    }
    fin.seekg(16);

    uint8_t image[MNIST_NINPUT];
    struct ggml_tensor * images = ggml_opt_dataset_data(dataset);
    float * buf = ggml_get_data_f32(images);

    GGML_ASSERT(images->ne[0] == MNIST_NINPUT);
    for (int64_t iex = 0; iex < images->ne[1]; ++iex) {
        fin.read((char *) image, sizeof(image));

        for (int64_t i = 0; i < MNIST_NINPUT; ++i) {
            buf[iex*MNIST_NINPUT + i] = image[i] / 255.0f; // Normalize to [0, 1]
        }
    }

    return true;
}

void mnist_image_print(FILE * stream, ggml_opt_dataset_t dataset, const int iex) {
    struct ggml_tensor * images = ggml_opt_dataset_data(dataset);
    GGML_ASSERT(images->ne[0] == MNIST_NINPUT);
    GGML_ASSERT(iex < images->ne[1]);
    const float * image = ggml_get_data_f32(images) + iex*MNIST_NINPUT;

    for (int64_t row = 0; row < MNIST_HW; row++) {
        for (int64_t col = 0; col < MNIST_HW; col++) {
            const int rgb = roundf(255.0f * image[row*MNIST_HW + col]);
#ifdef _WIN32
            fprintf(stream, "%s", rgb >= 220 ? "##" : "__");                // Represented via text.
#else
            fprintf(stream, "\033[48;2;%d;%d;%dm  \033[0m", rgb, rgb, rgb); // Represented via colored blocks.
#endif // _WIN32
        }
        fprintf(stream, "\n");
    }
}

bool mnist_label_load(const std::string & fname, ggml_opt_dataset_t dataset) {
    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "failed to open labels file %s\n", fname.c_str());
        return 0;
    }
    fin.seekg(8);

    uint8_t label;
    struct ggml_tensor * labels = ggml_opt_dataset_labels(dataset);
    float * buf = ggml_get_data_f32(labels);

    GGML_ASSERT(labels->ne[0] == MNIST_NCLASSES);
    for (int64_t iex = 0; iex < labels->ne[1]; ++iex) {
        fin.read((char *) &label, sizeof(label));

        for (int64_t i = 0; i < MNIST_NCLASSES; ++i) {
            buf[iex*MNIST_NCLASSES + i] = i == label ? 1.0f : 0.0f;
        }
    }

    return true;
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

mnist_model mnist_model_init_from_file(const std::string & fname, const std::string & backend, const int nbatch_logical, const int nbatch_physical) {
    mnist_model model(backend, nbatch_logical, nbatch_physical);
    fprintf(stderr, "%s: loading model weights from '%s'\n", __func__, fname.c_str());

    struct gguf_context * ctx;
    {
        struct gguf_init_params params = {
            /*.no_alloc   =*/ true,
            /*.ctx        =*/ &model.ctx_gguf,
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
    } else if (model.arch == "mnist-cnn") {
        model.conv1_kernel = ggml_get_tensor(model.ctx_gguf, "conv1.kernel");
        GGML_ASSERT(model.conv1_kernel->type == GGML_TYPE_F32);
        GGML_ASSERT(model.conv1_kernel->ne[0] == 3);
        GGML_ASSERT(model.conv1_kernel->ne[1] == 3);
        GGML_ASSERT(model.conv1_kernel->ne[2] == 1);
        GGML_ASSERT(model.conv1_kernel->ne[3] == MNIST_CNN_NCB);

        model.conv1_bias = ggml_get_tensor(model.ctx_gguf, "conv1.bias");
        GGML_ASSERT(model.conv1_bias->type == GGML_TYPE_F32);
        GGML_ASSERT(model.conv1_bias->ne[0] == 1);
        GGML_ASSERT(model.conv1_bias->ne[1] == 1);
        GGML_ASSERT(model.conv1_bias->ne[2] == MNIST_CNN_NCB);
        GGML_ASSERT(model.conv1_bias->ne[3] == 1);

        model.conv2_kernel = ggml_get_tensor(model.ctx_gguf, "conv2.kernel");
        GGML_ASSERT(model.conv2_kernel->type == GGML_TYPE_F32);
        GGML_ASSERT(model.conv2_kernel->ne[0] == 3);
        GGML_ASSERT(model.conv2_kernel->ne[1] == 3);
        GGML_ASSERT(model.conv2_kernel->ne[2] == MNIST_CNN_NCB);
        GGML_ASSERT(model.conv2_kernel->ne[3] == MNIST_CNN_NCB*2);

        model.conv2_bias = ggml_get_tensor(model.ctx_gguf, "conv2.bias");
        GGML_ASSERT(model.conv2_bias->type == GGML_TYPE_F32);
        GGML_ASSERT(model.conv2_bias->ne[0] == 1);
        GGML_ASSERT(model.conv2_bias->ne[1] == 1);
        GGML_ASSERT(model.conv2_bias->ne[2] == MNIST_CNN_NCB*2);
        GGML_ASSERT(model.conv2_bias->ne[3] == 1);

        model.dense_weight = ggml_get_tensor(model.ctx_gguf, "dense.weight");
        GGML_ASSERT(model.dense_weight->type == GGML_TYPE_F32);
        GGML_ASSERT(model.dense_weight->ne[0] == (MNIST_HW/4)*(MNIST_HW/4)*(MNIST_CNN_NCB*2));
        GGML_ASSERT(model.dense_weight->ne[1] == MNIST_NCLASSES);
        GGML_ASSERT(model.dense_weight->ne[2] == 1);
        GGML_ASSERT(model.dense_weight->ne[3] == 1);

        model.dense_bias = ggml_get_tensor(model.ctx_gguf, "dense.bias");
        GGML_ASSERT(model.dense_bias->type == GGML_TYPE_F32);
        GGML_ASSERT(model.dense_bias->ne[0] == MNIST_NCLASSES);
        GGML_ASSERT(model.dense_bias->ne[1] == 1);
        GGML_ASSERT(model.dense_bias->ne[2] == 1);
        GGML_ASSERT(model.dense_bias->ne[3] == 1);
    } else {
        fprintf(stderr, "%s: unknown model arch: %s\n", __func__, model.arch.c_str());
    }

    model.buf_gguf = ggml_backend_alloc_ctx_tensors(model.ctx_gguf, model.backends[0]);

    if(!load_from_gguf(fname.c_str(), model.ctx_gguf, ctx)) {
        fprintf(stderr, "%s: loading weights from %s failed\n", __func__, fname.c_str());
        exit(1);
    }

    // The space in ctx_gguf exactly fits the model weights,
    // the images (which also need to be statically allocated) need to be put in a different context.

    model.images = ggml_new_tensor_2d(model.ctx_static, GGML_TYPE_F32, MNIST_NINPUT, MNIST_NBATCH_PHYSICAL);
    ggml_set_name(model.images, "images");
    ggml_set_input(model.images);

    model.buf_static = ggml_backend_alloc_ctx_tensors(model.ctx_static, model.backends[0]);

    fprintf(stderr, "%s: successfully loaded weights from %s\n", __func__, fname.c_str());
    return model;
}

mnist_model mnist_model_init_random(const std::string & arch, const std::string & backend, const int nbatch_logical, const int nbatch_physical) {
    mnist_model model(backend, nbatch_logical, nbatch_physical);
    model.arch = arch;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> nd{0.0f, 1e-2f};
    std::vector<ggml_tensor *> init_tensors;

    if (model.arch == "mnist-fc") {
        fprintf(stderr, "%s: initializing random weights for a fully connected model\n", __func__);

        model.fc1_weight = ggml_new_tensor_2d(model.ctx_static, GGML_TYPE_F32, MNIST_NINPUT,  MNIST_NHIDDEN);
        model.fc1_bias   = ggml_new_tensor_1d(model.ctx_static, GGML_TYPE_F32,                MNIST_NHIDDEN);
        model.fc2_weight = ggml_new_tensor_2d(model.ctx_static, GGML_TYPE_F32, MNIST_NHIDDEN, MNIST_NCLASSES);
        model.fc2_bias   = ggml_new_tensor_1d(model.ctx_static, GGML_TYPE_F32,                MNIST_NCLASSES);

        ggml_set_name(model.fc1_weight, "fc1.weight");
        ggml_set_name(model.fc1_bias,   "fc1.bias");
        ggml_set_name(model.fc2_weight, "fc2.weight");
        ggml_set_name(model.fc2_bias,   "fc2.bias");

        init_tensors.push_back(model.fc1_weight);
        init_tensors.push_back(model.fc1_bias);
        init_tensors.push_back(model.fc2_weight);
        init_tensors.push_back(model.fc2_bias);
    } else if (model.arch == "mnist-cnn") {
        model.conv1_kernel = ggml_new_tensor_4d(model.ctx_static, GGML_TYPE_F32, 3, 3, 1, MNIST_CNN_NCB);
        model.conv1_bias   = ggml_new_tensor_3d(model.ctx_static, GGML_TYPE_F32, 1, 1,    MNIST_CNN_NCB);
        model.conv2_kernel = ggml_new_tensor_4d(model.ctx_static, GGML_TYPE_F32, 3, 3, MNIST_CNN_NCB, MNIST_CNN_NCB*2);
        model.conv2_bias   = ggml_new_tensor_3d(model.ctx_static, GGML_TYPE_F32, 1, 1,                MNIST_CNN_NCB*2);
        model.dense_weight = ggml_new_tensor_2d(model.ctx_static, GGML_TYPE_F32, (MNIST_HW/4)*(MNIST_HW/4)*(MNIST_CNN_NCB*2), MNIST_NCLASSES);
        model.dense_bias   = ggml_new_tensor_1d(model.ctx_static, GGML_TYPE_F32, MNIST_NCLASSES);

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

    model.images = ggml_new_tensor_2d(model.ctx_static, GGML_TYPE_F32, MNIST_NINPUT, MNIST_NBATCH_PHYSICAL);
    ggml_set_name(model.images, "images");
    ggml_set_input(model.images);

    model.buf_static = ggml_backend_alloc_ctx_tensors(model.ctx_static, model.backends[0]);

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

void mnist_model_build(mnist_model & model) {
    if (model.arch == "mnist-fc") {
        ggml_set_param(model.ctx_compute, model.fc1_weight);
        ggml_set_param(model.ctx_compute, model.fc1_bias);
        ggml_set_param(model.ctx_compute, model.fc2_weight);
        ggml_set_param(model.ctx_compute, model.fc2_bias);

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

        struct ggml_tensor * images_2D = ggml_reshape_4d(model.ctx_compute, model.images, MNIST_HW, MNIST_HW, 1, model.images->ne[1]);

        struct ggml_tensor * conv1_out = ggml_relu(model.ctx_compute, ggml_add(model.ctx_compute,
            ggml_conv_2d(model.ctx_compute, model.conv1_kernel, images_2D, 1, 1, 1, 1, 1, 1),
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
}

ggml_opt_result_t mnist_model_eval(mnist_model & model, ggml_opt_dataset_t dataset) {
    ggml_opt_result_t result = ggml_opt_result_init();

    ggml_opt_params params = ggml_opt_default_params(model.backend_sched, model.ctx_compute, model.images, model.logits, GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    params.build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params);

    {
        const int64_t t_start_us = ggml_time_us();

        ggml_opt_epoch(opt_ctx, dataset, nullptr, result, /*idata_split =*/ 0, nullptr, nullptr);

        const int64_t t_total_us = ggml_time_us() - t_start_us;
        const double t_total_ms = 1e-3*t_total_us;
        const int nex = ggml_opt_dataset_data(dataset)->ne[1];
        fprintf(stderr, "%s: model evaluation on %d images took %.2lf ms, %.2lf us/image\n",
                __func__, nex, t_total_ms, (double) t_total_us/nex);
    }

    ggml_opt_free(opt_ctx);

    return result;
}

void mnist_model_train(mnist_model & model, ggml_opt_dataset_t dataset, const int nepoch, const float val_split) {
    ggml_opt_fit(model.backend_sched, model.ctx_compute, model.images, model.logits, dataset,
        GGML_OPT_LOSS_TYPE_CROSS_ENTROPY, ggml_opt_get_default_optimizer_params, nepoch, model.nbatch_logical, val_split, false);
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

#ifdef __cplusplus
extern "C" {
#endif

int wasm_eval(uint8_t * digitPtr) {
    std::vector<float> digit(digitPtr, digitPtr + MNIST_NINPUT);

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(MNIST_NINPUT, MNIST_NCLASSES, 1, 1);
    struct ggml_tensor * data = ggml_opt_dataset_data(dataset);
    memcpy(data->data, digitPtr, ggml_nbytes(data));
    ggml_set_zero(ggml_opt_dataset_labels(dataset)); // The labels are not needed.

    mnist_model model = mnist_model_init_from_file("mnist-f32.gguf", "CPU", /*nbatch_logical =*/ 1, /*nbatch_physical =*/ 1);
    mnist_model_build(model);
    ggml_opt_result_t result = mnist_model_eval(model, dataset);

    int32_t pred;
    ggml_opt_result_pred(result, &pred);

    return pred;
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
