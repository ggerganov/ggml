#include <algorithm>
#include <cstdint>
#include <random>
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

struct mnist_dataset {
    struct ggml_context * ctx;
    struct ggml_tensor  * data;
    struct ggml_tensor  * labels;

    int64_t nex;
    int64_t shard_size;
    size_t  nbs_data;
    size_t  nbs_labels;

    std::vector<int64_t> permutation;
    std::mt19937 rng;

    mnist_dataset(const int64_t nex, const int64_t shard_size) : nex(nex), shard_size(shard_size) {
        const size_t nbytes_images = nex*MNIST_NINPUT  *sizeof(float) + ggml_tensor_overhead();
        const size_t nbytes_labels = nex*MNIST_NCLASSES*sizeof(float) + ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ nbytes_images + nbytes_labels,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        ctx = ggml_init(params);

        data   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, MNIST_HW, MNIST_HW, nex);
        labels = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MNIST_NCLASSES,     nex);

        nbs_data   = ggml_nbytes(data)   * shard_size/nex;
        nbs_labels = ggml_nbytes(labels) * shard_size/nex;

        permutation.resize(nex/shard_size);
        for (size_t i = 0; i < permutation.size(); ++i) {
            permutation[i] = i;
        }
    }

    ~mnist_dataset() {
        ggml_free(ctx);
    }

    void shuffle(const size_t ishard_max) {
        if (ishard_max < permutation.size()) {
            std::shuffle(permutation.begin(), permutation.begin() + ishard_max, rng);
            return;
        }
        std::shuffle(permutation.begin(), permutation.end(), rng);
    }

    void get_batch(struct ggml_tensor * data_batch, struct ggml_tensor * labels_batch, const int64_t ibatch) {
        const int64_t shards_per_batch = ggml_nbytes(data_batch) / nbs_data;
        for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
            const int64_t ishard = permutation[ibatch*shards_per_batch + ishard_batch];

            ggml_backend_tensor_set(data_batch,   (const char *)   data->data + ishard*nbs_data,   ishard_batch*nbs_data,   nbs_data);
            ggml_backend_tensor_set(labels_batch, (const char *) labels->data + ishard*nbs_labels, ishard_batch*nbs_labels, nbs_labels);
        }
    }
};

struct mnist_model {
    std::string arch;
    ggml_backend_t backend;
    int nbatch_logical;
    int nbatch_physical;

    struct ggml_tensor  * images    = nullptr;
    struct ggml_tensor  * labels    = nullptr;
    struct ggml_tensor  * logits    = nullptr;
    struct ggml_tensor  * probs     = nullptr;
    struct ggml_tensor  * loss      = nullptr;
    struct ggml_tensor  * pred      = nullptr;
    struct ggml_tensor  * acc_count = nullptr;

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

    struct ggml_context * ctx_weight  = nullptr;
    struct ggml_context * ctx_compute = nullptr;
    ggml_backend_buffer_t buf_weight  = nullptr;
    ggml_backend_buffer_t buf_compute = nullptr;

    mnist_model(const std::string & backend_name) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name.c_str());
        if (dev == nullptr) {
            fprintf(stderr, "%s: ERROR: backend %s not found, available:\n", __func__, backend_name.c_str());
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t this_dev = ggml_backend_dev_get(i);
                fprintf(stderr, "  - %s (%s)\n", ggml_backend_dev_name(this_dev), ggml_backend_dev_description(this_dev));
            }
            exit(1);
        }

        fprintf(stderr, "%s: using %s (%s) backend\n", __func__, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));

        backend = ggml_backend_dev_init(dev, NULL);
        if (ggml_backend_is_cpu(backend)) {
            const int ncores_logical = std::thread::hardware_concurrency();
            ggml_backend_cpu_set_n_threads(backend, std::min(ncores_logical, (ncores_logical + 4)/2));
        }

        {
            const size_t size_meta = 1024*ggml_tensor_overhead();
            struct ggml_init_params params = {
                /*.mem_size   =*/ size_meta,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ctx_weight = ggml_init(params);
        }

        {
            // The compute context needs a total of 3 compute graphs: forward pass + backwards pass (with/without optimizer step).
            const size_t size_meta = GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead();
            struct ggml_init_params params = {
                /*.mem_size   =*/ size_meta,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ctx_compute = ggml_init(params);
        }
    }

    ~mnist_model() {
        ggml_free(ctx_weight);
        ggml_free(ctx_compute);

        ggml_backend_buffer_free(buf_weight);
        ggml_backend_buffer_free(buf_compute);
        ggml_backend_free(backend);
    }
};

struct mnist_eval_result {
    bool success = false;

    std::vector<float>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;
    int64_t              ntotal   = 0;
};

bool mnist_image_load(const std::string & fname, mnist_dataset & dataset);
void mnist_image_print(FILE * f, mnist_dataset & dataset, const int iex);
bool mnist_label_load(const std::string & fname, mnist_dataset & dataset);

mnist_eval_result mnist_graph_eval(const std::string & fname, const float * images, const float * labels, const int nex, const int nthreads);

mnist_model       mnist_model_init_from_file(const std::string & fname, const std::string & backend);
mnist_model       mnist_model_init_random(const std::string & arch, const std::string & backend);
void              mnist_model_build(mnist_model & model, const int nbatch_logical, const int nbatch_physical);
mnist_eval_result mnist_model_eval(mnist_model & model, mnist_dataset & dataset);
void              mnist_model_train(mnist_model & model, mnist_dataset & dataset, const int nepoch, const float val_split);
void              mnist_model_save(mnist_model & model, const std::string & fname);

std::pair<double, double> mnist_loss(const mnist_eval_result & result);
std::pair<double, double> mnist_accuracy(const mnist_eval_result & result);
