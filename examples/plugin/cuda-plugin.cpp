#include "model.hpp"

#include <vector>
#include <iostream>

#include <ggml-cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    // init cuda
    int device_id = 0;
    cudaSetDevice(device_id);
    cublasHandle_t cublas_handle = nullptr;
    cublasCreate(&cublas_handle);
    cudaStream_t cuda_stream = nullptr;
    cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);

    // create plugin backend
    auto backend = ggml_backend_cuda_init_plugin(device_id, cublas_handle, cuda_stream);

    // init weights
    std::vector<float> weights_data;
    for (int i = 0; i < 10; ++i) {
        weights_data.push_back(float(i));
    }

    void* weights = nullptr;
    cudaMallocAsync(&weights, data_size(weights_data), cuda_stream);
    cudaMemcpyAsync(weights, weights_data.data(), data_size(weights_data), cudaMemcpyHostToDevice, cuda_stream);

    // create model with weights
    model m(backend, weights_data.size(), GGML_TYPE_F32, weights);

    // init input and output data
    std::vector<float> input_data;
    for (size_t i = 0; i < weights_data.size(); ++i) {
        input_data.push_back(float(i) / 10);
    }

    std::vector<float> output_data(input_data.size());

    void* input = nullptr;
    cudaMallocAsync(&input, data_size(input_data), cuda_stream);
    cudaMemcpyAsync(input, input_data.data(), data_size(input_data), cudaMemcpyHostToDevice, cuda_stream);

    void* output = nullptr;
    cudaMallocAsync(&output, data_size(output_data), cuda_stream);

    // compute with cuda pointers
    m.compute(output, input);

    // get data back from cuda pointers
    cudaMemcpyAsync(output_data.data(), output, data_size(output_data), cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);

    ggml_backend_free(backend);

    // print result
    std::cout << "[";
    for (auto o : output_data) {
        std::cout << o << ", ";
    }
    std::cout << "]\n";

    return 0;
}
