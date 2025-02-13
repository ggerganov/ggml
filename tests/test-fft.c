#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#include "ggml.h"
#include "ggml-cpu.h"

#define N_SAMPLES 16  // Must be power of 2

// Helper function to generate a simple test signal
void generate_test_signal(float * signal, int n) {
    // Generate a simple sinusoidal signal
    for (int i = 0; i < n; i++) {
        signal[i] = sinf(2.0f * M_PI * i / n) + 0.5f * sinf(4.0f * M_PI * i / n);
    }
}

// Helper function to compare arrays with tolerance
bool compare_arrays(float * a, float * b, int n, float tolerance) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > tolerance) {
            printf("Mismatch at index %d: %f != %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, const char ** argv) {
    struct ggml_init_params params = {
        .mem_size   = 128*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    // initialize the backend
    struct ggml_context * ctx = ggml_init(params);

    // Create test signal
    float input_signal[N_SAMPLES];
    float output_signal[N_SAMPLES];
    generate_test_signal(input_signal, N_SAMPLES);

    // Create tensors
    struct ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_SAMPLES);
    struct ggml_tensor * fft_result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * N_SAMPLES);
    struct ggml_tensor * ifft_result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_SAMPLES);

    // Copy input signal to tensor
    memcpy(input->data, input_signal, N_SAMPLES * sizeof(float));

    // Create compute graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    struct ggml_cgraph * gb = ggml_new_graph(ctx);

    // Perform FFT
    fft_result = ggml_fft(ctx, input);
    ggml_build_forward_expand(gf, fft_result);

    // Perform IFFT
    ifft_result = ggml_ifft(ctx, fft_result);
    ggml_build_forward_expand(gb, ifft_result);

    // Compute the graphs
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    ggml_graph_compute_with_ctx(ctx, gb, 1);

    // Copy result back
    memcpy(output_signal, ifft_result->data, N_SAMPLES * sizeof(float));

    // Compare input and output
    const float tolerance = 1e-5f;
    bool success = compare_arrays(input_signal, output_signal, N_SAMPLES, tolerance);

    if (success) {
        printf("FFT/IFFT test passed! Signal was correctly reconstructed within tolerance %f\n", tolerance);
    } else {
        printf("FFT/IFFT test failed! Signal reconstruction error exceeded tolerance %f\n", tolerance);
        
        // Print signals for comparison
        printf("\nOriginal signal:\n");
        for (int i = 0; i < N_SAMPLES; i++) {
            printf("%f ", input_signal[i]);
        }
        printf("\n\nReconstructed signal:\n");
        for (int i = 0; i < N_SAMPLES; i++) {
            printf("%f ", output_signal[i]);
        }
        printf("\n");
    }

    ggml_free(ctx);

    return success ? 0 : 1;
}
