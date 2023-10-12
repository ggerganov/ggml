#include "model.hpp"

#include <ggml-backend.h>

#include <vector>
#include <iostream>

int main() {
    auto backend = ggml_backend_cpu_init();

    std::vector<float> weights_data;
    for (int i = 0; i < 10; ++i) {
        weights_data.push_back(float(i));
    }

    void* weights = weights_data.data();

    model m(backend, weights_data.size(), GGML_TYPE_F32, weights);

    std::vector<float> input_data;
    for (size_t i = 0; i < weights_data.size(); ++i) {
        input_data.push_back(float(i) / 10);
    }

    std::vector<float> output_data(input_data.size());

    void* input = input_data.data();
    void* output = output_data.data();

    m.compute(output, input);

    ggml_backend_free(backend);

    std::cout << "[";
    for (auto o : output_data) {
        std::cout << o << ", ";
    }
    std::cout << "]\n";

    return 0;
}
