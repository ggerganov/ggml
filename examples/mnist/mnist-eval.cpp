#include "ggml.h"
#include "ggml-opt.h"

#include "mnist-common.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

int main(int argc, char ** argv) {
    srand(time(NULL));
    ggml_time_init();

    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s mnist-fc-f32.gguf data/MNIST/raw/t10k-images-idx3-ubyte data/MNIST/raw/t10k-labels-idx1-ubyte [CPU/CUDA0]\n", argv[0]);
        exit(1);
    }

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(MNIST_NINPUT, MNIST_NCLASSES, MNIST_NTEST, MNIST_NBATCH_PHYSICAL);

    if (!mnist_image_load(argv[2], dataset)) {
        return 1;
    }
    if (!mnist_label_load(argv[3], dataset)) {
        return 1;
    }

    const int iex = rand() % MNIST_NTEST;
    mnist_image_print(stdout, dataset, iex);

    const std::string backend = argc >= 5 ? argv[4] : "";

    const int64_t t_start_us = ggml_time_us();
    mnist_model model = mnist_model_init_from_file(argv[1], backend, MNIST_NBATCH_LOGICAL, MNIST_NBATCH_PHYSICAL);
    mnist_model_build(model);
    const int64_t t_load_us = ggml_time_us() - t_start_us;
    fprintf(stdout, "%s: loaded model in %.2lf ms\n", __func__, t_load_us / 1000.0);

    ggml_opt_result_t result_eval = mnist_model_eval(model, dataset);

    std::vector<int32_t> pred(MNIST_NTEST);
    ggml_opt_result_pred(result_eval, pred.data());
    fprintf(stdout, "%s: predicted digit is %d\n", __func__, pred[iex]);

    double loss;
    double loss_unc;
    ggml_opt_result_loss(result_eval, &loss, &loss_unc);
    fprintf(stdout, "%s: test_loss=%.6lf+-%.6lf\n", __func__, loss, loss_unc);

    double accuracy;
    double accuracy_unc;
    ggml_opt_result_accuracy(result_eval, &accuracy, &accuracy_unc);
    fprintf(stdout, "%s: test_acc=%.2lf+-%.2lf%%\n", __func__, 100.0*accuracy, 100.0*accuracy_unc);

    ggml_opt_result_free(result_eval);

    return 0;
}
