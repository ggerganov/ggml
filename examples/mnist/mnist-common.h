#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#define MNIST_NTRAIN          60000
#define MNIST_NTEST           10000
#define MNIST_NBATCH_LOGICAL   1000
#define MNIST_NBATCH_PHYSICAL   500

static_assert(MNIST_NBATCH_LOGICAL % MNIST_NBATCH_PHYSICAL == 0, "MNIST_NBATCH_LOGICAL % MNIST_NBATCH_PHYSICAL != 0");
static_assert(MNIST_NTRAIN % MNIST_NBATCH_LOGICAL == 0, "MNIST_NTRAIN % MNIST_NBATCH_LOGICAL != 0");
static_assert(MNIST_NTEST  % MNIST_NBATCH_LOGICAL == 0, "MNIST_NTRAIN % MNIST_NBATCH_LOGICAL != 0");

#define MNIST_HW       28
#define MNIST_NINPUT   (MNIST_HW*MNIST_HW)
#define MNIST_NCLASSES 10

#define MNIST_NHIDDEN  500

// NCB = number of channels base
#define MNIST_CNN_NCB 8

struct mnist_model {
    std::string arch;
    ggml_backend_t backend;
    int nbatch_logical;
    int nbatch_physical;

    struct ggml_tensor  * images = nullptr;
    struct ggml_tensor  * labels = nullptr;
    struct ggml_tensor  * logits = nullptr;
    struct ggml_tensor  * probs  = nullptr;
    struct ggml_tensor  * loss   = nullptr;

    struct ggml_tensor * fc1_weight = nullptr;
    struct ggml_tensor * fc1_bias   = nullptr;
    struct ggml_tensor * fc2_weight = nullptr;
    struct ggml_tensor * fc2_bias   = nullptr;

    struct ggml_tensor * conv1_kernel = nullptr;
    struct ggml_tensor * conv1_bias   = nullptr;
    struct ggml_tensor * conv2_kernel = nullptr;
    struct ggml_tensor * conv2_bias   = nullptr;
    struct ggml_tensor * dense_weight = nullptr;
    struct ggml_tensor * dense_bias   = nullptr;

    static const size_t size_weight  = 100 *      1024*1024;
    static const size_t size_compute =   1 * 1024*1024*1024;

    void                * buf_weight  = nullptr;
    struct ggml_context * ctx_weight  = nullptr;
    void                * buf_compute = nullptr;
    struct ggml_context * ctx_compute = nullptr;
    ggml_backend_buffer_t buf_backend = nullptr;
    ggml_backend_buffer_t buf_weightt = nullptr;

    mnist_model(const std::string & backend_name) {
        const size_t backend_index = ggml_backend_reg_find_by_name(backend_name.c_str());
        if (backend_index == SIZE_MAX) {
            fprintf(stderr, "%s: ERROR: backend %s not found, available:\n", __func__, backend_name.c_str());
            for (size_t i = 0; i < ggml_backend_reg_get_count(); ++i) {
                fprintf(stderr, "  - %s\n", ggml_backend_reg_get_name(i));
            }
            exit(1);
        }

        fprintf(stderr, "%s: using %s backend\n", __func__, backend_name.c_str());
        backend = ggml_backend_reg_init_backend(backend_index, nullptr);
        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, std::thread::hardware_concurrency());
        }

        buf_weight = malloc(size_weight);
        {
            struct ggml_init_params params = {
                /*.mem_size   =*/ size_weight,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ctx_weight = ggml_init(params);
        }

        buf_compute = malloc(size_compute);
        {
            struct ggml_init_params params = {
                /*.mem_size   =*/ size_compute,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ctx_compute = ggml_init(params);
        }
    }

    ~mnist_model() {
        ggml_free(ctx_weight);
        ggml_free(ctx_compute);

        free(buf_weight);
        free(buf_compute);

        ggml_backend_buffer_free(buf_weightt);
        ggml_backend_buffer_free(buf_backend);
        ggml_backend_free(backend);
    }
};

struct mnist_eval_result {
    bool success = false;

    std::vector<float>   loss;
    std::vector<int32_t> pred;
};

bool mnist_image_load(const std::string & fname, float * buf, const int nex);
void mnist_image_print(FILE * f, const float * image);
bool mnist_label_load(const std::string & fname, float * buf, const int nex);

mnist_eval_result mnist_graph_eval(const std::string & fname, const float * images, const float * labels, const int nex, const int nthreads);

mnist_model       mnist_model_init_from_file(const std::string & fname, const std::string & backend);
mnist_model       mnist_model_init_random(const std::string & arch, const std::string & backend);
void              mnist_model_build(mnist_model & model, const int nbatch_logical, const int nbatch_physical);
mnist_eval_result mnist_model_eval(mnist_model & model, const float * images, const float * labels, const int nex);
void              mnist_model_train(mnist_model & model, const float * images, const float * labels, const int nex);
void              mnist_model_save(mnist_model & model, const std::string & fname);

std::pair<double, double> mnist_loss(const mnist_eval_result & result);
std::pair<double, double> mnist_accuracy(const mnist_eval_result & result, const float * labels);
