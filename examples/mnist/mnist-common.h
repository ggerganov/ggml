#include <string>
#include <vector>

#define MNIST_NTRAIN 60000
#define MNIST_NTEST  10000
#define MNIST_NBATCH 1000

static_assert(MNIST_NTRAIN % MNIST_NBATCH == 0, "MNIST_NTRAIN % MNIST_BATCH != 0");
static_assert(MNIST_NTEST  % MNIST_NBATCH == 0, "MNIST_NTRAIN % MNIST_BATCH != 0");

#define MNIST_NINPUT   784
#define MNIST_NHIDDEN  500
#define MNIST_NCLASSES 10

struct mnist_model {
    int nbatch;

    struct ggml_context * ctx_gguf;
    struct ggml_tensor  * fc1_weight;
    struct ggml_tensor  * fc1_bias;
    struct ggml_tensor  * fc2_weight;
    struct ggml_tensor  * fc2_bias;

    struct ggml_context * ctx_eval;

    void                * buf_compute;
    struct ggml_context * ctx_compute;
    struct ggml_tensor  * images;
    struct ggml_tensor  * labels;
    struct ggml_tensor  * logits;
    struct ggml_tensor  * probs;
    struct ggml_tensor  * loss;
};

struct mnist_eval_result {
    bool success = false;

    std::vector<float>   loss;
    std::vector<int32_t> pred;
};

bool mnist_image_load(const std::string & fname, float * buf, const int nex);
void mnist_image_print(FILE * f, const float * image);
bool mnist_label_load(const std::string & fname, float * buf, const int nex);

mnist_eval_result mnist_graph_eval(const std::string & fname, const float * images, const float * labels, const int nex);

mnist_model       mnist_model_init(const std::string & fname, const int nbatch);
void              mnist_model_free(mnist_model & model);
mnist_eval_result mnist_model_eval(const mnist_model & model, const float * images, const float * labels, const int nex);
void              mnist_model_train(const float * images, const float * labels, const int nex, mnist_model & model);
void              mnist_model_save(const std::string & fname, mnist_model & model);

std::pair<double, double> mnist_loss(const mnist_eval_result & result);
std::pair<double, double> mnist_accuracy(const mnist_eval_result & result, const float * labels);
