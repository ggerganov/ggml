#include <string>
#include <vector>

#include "ggml.h"

#define MNIST_NTRAIN 60000
#define MNIST_NTEST  10000
#define MNIST_NBATCH 500

static_assert(MNIST_NTRAIN % MNIST_NBATCH == 0, "MNIST_NTRAIN % MNIST_BATCH != 0");
static_assert(MNIST_NTEST  % MNIST_NBATCH == 0, "MNIST_NTRAIN % MNIST_BATCH != 0");

#define MNIST_HW       28
#define MNIST_NINPUT   (MNIST_HW*MNIST_HW)
#define MNIST_NCLASSES 10

#define MNIST_NHIDDEN  500

// NCB = number of channels base
#define MNIST_CNN_NCB 8

struct mnist_model {
    std::string arch;
    int nbatch;

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

    mnist_model() {
        buf_weight = malloc(size_weight);
        {
            struct ggml_init_params params = {
                /*.mem_size   =*/ size_weight,
                /*.mem_buffer =*/ buf_weight,
                /*.no_alloc   =*/ false,
            };
            ctx_weight = ggml_init(params);
        }

        buf_compute = malloc(size_compute);
        {
            struct ggml_init_params params = {
                /*.mem_size   =*/ size_compute,
                /*.mem_buffer =*/ buf_compute,
                /*.no_alloc   =*/ false,
            };
            ctx_compute = ggml_init(params);
        }
    }

    ~mnist_model() {
        ggml_free(ctx_weight);
        ggml_free(ctx_compute);

        free(buf_weight);
        free(buf_compute);
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

mnist_model       mnist_model_init_from_file(const std::string & fname);
mnist_model       mnist_model_init_random(const std::string & arch);
void              mnist_model_build(mnist_model & model, const int nbatch);
mnist_eval_result mnist_model_eval(const mnist_model & model, const float * images, const float * labels, const int nex, const int nthreads);
void              mnist_model_train(mnist_model & model, const float * images, const float * labels, const int nex, const int nthreads);
void              mnist_model_save(mnist_model & model, const std::string & fname);

std::pair<double, double> mnist_loss(const mnist_eval_result & result);
std::pair<double, double> mnist_accuracy(const mnist_eval_result & result, const float * labels);
