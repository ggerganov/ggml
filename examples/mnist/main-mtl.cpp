// Use a pre-generated MNIST compute graph for inference on the M1 GPU via MPS
//
// You can generate a compute graph using the "mnist" tool:
//
// $ ./bin/mnist ./models/mnist/ggml-model-f32.bin ../examples/mnist/models/mnist/t10k-images.idx3-ubyte
//
// This command creates the "mnist.ggml" file, which contains the generated compute graph.
// Now, you can re-use the compute graph on the GPU with the "mnist-mtl" tool:
//
// $ ./bin/mnist-mtl ./models/mnist/mnist.ggml ../examples/mnist/models/mnist/t10k-images.idx3-ubyte
//

#include "ggml/ggml.h"

#include "main-mtl.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <vector>

// evaluate the MNIST compute graph
//
//   - fname_cgraph: path to the compute graph
//   - digit:        784 pixel values
//
// returns 0 - 9 prediction
int mnist_eval(
        const char * fname_cgraph,
        std::vector<float> digit
        ) {
    // load the compute graph
    struct ggml_context * ctx_data = NULL;
    struct ggml_context * ctx_eval = NULL;

    struct ggml_cgraph gf = ggml_graph_import(fname_cgraph, &ctx_data, &ctx_eval);

    // allocate work context
    static size_t buf_size = 128ull*1024*1024; // TODO
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx_work = ggml_init(params);

    // this allocates all Metal resources and memory buffers
    auto ctx_mtl = mnist_mtl_init(ctx_data, ctx_eval, ctx_work, &gf);

    int prediction = -1;

    for (int i = 0; i < 1; ++i) {
        struct ggml_tensor * input = ggml_graph_get_tensor(&gf, "input");

        if (i % 2 == 0) {
            memcpy(input->data, digit.data(), ggml_nbytes(input));
        } else {
            memset(input->data, 0, ggml_nbytes(input));
        }

        // the actual inference happens here
        prediction = mnist_mtl_eval(ctx_mtl, &gf);
    }

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

    const int prediction = mnist_eval(argv[1], digit);

    fprintf(stdout, "%s: predicted digit is %d\n", __func__, prediction);

    return 0;
}
