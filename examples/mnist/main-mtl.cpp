// Use a pre-generated MNIST compute graph for inference on the M1 GPU via MPS
//

#include "ggml/ggml.h"

#include "main-mtl.h"

#include "common-ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>

// evaluate the MNIST compute graph
//
//   - fname_cgraph: path to the compute graph
//   - n_threads:    number of threads to use
//   - digit:        784 pixel values
//
// returns 0 - 9 prediction
int mnist_eval(
        const char * fname_cgraph,
        const int n_threads,
        std::vector<float> digit
        ) {
    // load the compute graph
    struct ggml_context * ctx_data = NULL;
    struct ggml_context * ctx_eval = NULL;

    struct ggml_cgraph gf = ggml_cgraph_import(fname_cgraph, &ctx_data, &ctx_eval);
    gf.n_threads = n_threads;

    // allocate eval context
    // needed during ggml_graph_compute() to allocate a work tensor
    static size_t buf_size = gf.work_size; // TODO
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = buf,
    };

    struct ggml_context * ctx_work = ggml_init(params);

    struct ggml_tensor * input = ggml_get_tensor_by_name(&gf, "input");
    memcpy(input->data, digit.data(), ggml_nbytes(input));

    //ggml_graph_compute(ctx_work, &gf);
    auto ctx_mtl = mnist_mtl_init(ctx_data, ctx_eval, ctx_work, &gf);
    const int prediction = mnist_mtl_eval(ctx_mtl, &gf);
    mnist_mtl_free(ctx_mtl);

    ggml_free(ctx_work);
    ggml_free(ctx_data);
    ggml_free(ctx_eval);

    return prediction;
}

int main(int argc, char ** argv) {
    srand(time(NULL));
    ggml_time_init();

    if (argc != 3) {
        fprintf(stderr, "Usage: %s models/mnist/mnist.ggml models/mnist/t10k-images.idx3-ubyte\n", argv[0]);
        exit(0);
    }

    uint8_t buf[784];
    std::vector<float> digit;

    // read a random digit from the test set
    {
        std::ifstream fin(argv[2], std::ios::binary);
        if (!fin) {
            fprintf(stderr, "%s: failed to open '%s'\n", __func__, argv[2]);
            return 1;
        }

        // seek to a random digit: 16-byte header + 28*28 * (random 0 - 10000)
        fin.seekg(16 + 784 * (rand() % 10000));
        fin.read((char *) &buf, sizeof(buf));
    }

    // render the digit in ASCII
    {
        digit.resize(sizeof(buf));

        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                fprintf(stderr, "%c ", (float)buf[row*28 + col] > 230 ? '*' : '_');
                digit[row*28 + col] = ((float)buf[row*28 + col]);
            }

            fprintf(stderr, "\n");
        }

        fprintf(stderr, "\n");
    }

    const int prediction = mnist_eval(argv[1], 1, digit);

    fprintf(stdout, "%s: predicted digit is %d\n", __func__, prediction);

    return 0;
}
