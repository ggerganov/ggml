#include "ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

// #define GGML_USE_CUBLAS

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

void filterTransform(float* output, float* kernel, int output_channels, int input_channels)
{
	int output_offset = input_channels * 16;
	int kernel_offset = input_channels * 9;
	for (int n_idx = 0; n_idx < output_channels; n_idx++)
	{
		int temp2u = n_idx * output_offset;
		int temp2f = n_idx * kernel_offset;
		for (int c_idx = 0; c_idx < input_channels; c_idx++)
		{
			int u_idx = c_idx * 16 + temp2u;
			int f_idx = c_idx * 9 + temp2f;

			float g1 = kernel[f_idx];
			float g2 = kernel[f_idx + 1];
			float g3 = kernel[f_idx + 2];
			float g4 = kernel[f_idx + 3];
			float g5 = kernel[f_idx + 4];
			float g6 = kernel[f_idx + 5];
			float g7 = kernel[f_idx + 6];
			float g8 = kernel[f_idx + 7];
			float g9 = kernel[f_idx + 8];

			output[u_idx] = g1;
			output[u_idx + 1] = (g1 + g2 + g3) / 2.f;
			output[u_idx + 2] = (g1 - g2 + g3) / 2.f;
			output[u_idx + 3] = g3;

			float temp1 = g1 + g4 + g7;
			float temp2 = g2 + g5 + g8;
			float temp3 = g3 + g6 + g9;

			output[u_idx + 4] = (temp1) / 2.f;
			output[u_idx + 5] = (temp1 + temp2 + temp3) / 4.f;
			output[u_idx + 6] = (temp1 - temp2 + temp3) / 4.f;
			output[u_idx + 7] = (temp3) / 2.f;

			float temp4 = g1 - g4 + g7;
			float temp5 = g2 - g5 + g8;
			float temp6 = g3 - g6 + g9;

			output[u_idx + 8] = (temp4) / 2.f;
			output[u_idx + 9] = (temp4 + temp5 + temp6) / 4.f;
			output[u_idx + 10] = (temp4 - temp5 + temp6) / 4.f;
			output[u_idx + 11] = (temp6) / 2.f;

			output[u_idx + 12] = g7;
			output[u_idx + 13] = (g7 + g8 + g9) / 2.f;
			output[u_idx + 14] = (g7 - g8 + g9) / 2.f;
			output[u_idx + 15] = g9;
		}
	}
}

void winogradConv2d(float* dst, float* image, float* kernel, int input_channels, int input_height, int input_width, int output_channels)
{
	float* U = (float*)malloc(output_channels * input_channels * 16 * sizeof(float)); // 64 bytes
	filterTransform(U, kernel, output_channels, input_channels);

	int output_h_z = input_height - 2;
	int output_w_z = input_width - 2;
	int temp_u = input_channels * 16;

    for (int row = 0; row < output_h_z; row += 2)
    {
        int row_idx1 = row * input_width;
        int row_idx2 = row_idx1 + input_width;
        int row_idx3 = row_idx2 + input_width;
        int row_idx4 = row_idx3 + input_width;
        int row_idxo1 = row * output_w_z;
        int row_idxo2 = row_idxo1 + output_w_z;

        for (int col = 0; col < output_w_z; col += 2)
        {
            for (int outch = 0; outch < output_channels; outch++)
            {
                int temp_u2 = outch * temp_u;
                int ot_idx1 = outch * output_h_z * output_w_z + col + row_idxo1;
                int ot_idx3 = outch * output_h_z * output_w_z + col + row_idxo2;
                float y1 = 0, y2 = 0, y3 = 0, y4 = 0;

                for (int inch = 0; inch < input_channels; inch++)
                {
                    int temp_ic = inch * input_height * input_width;
                    int u_idx = inch * 16 + temp_u2; // U idex

                    int t_idx1 = temp_ic + row_idx1 + col;
                    int t_idx2 = temp_ic + row_idx2 + col;
                    int t_idx3 = temp_ic + row_idx3 + col;
                    int t_idx4 = temp_ic + row_idx4 + col;

                    float d1 = image[t_idx1];
                    float d2 = image[t_idx1 + 1];
                    float d3 = image[t_idx1 + 2];
                    float d4 = image[t_idx1 + 3];

                    float d5 = image[t_idx2];
                    float d6 = image[t_idx2 + 1];
                    float d7 = image[t_idx2 + 2];
                    float d8 = image[t_idx2 + 3];

                    float d9 = image[t_idx3];
                    float d10 = image[t_idx3 + 1];
                    float d11 = image[t_idx3 + 2];
                    float d12 = image[t_idx3 + 3];

                    float d13 = image[t_idx4];
                    float d14 = image[t_idx4 + 1];
                    float d15 = image[t_idx4 + 2];
                    float d16 = image[t_idx4 + 3];

                    float dd1 = d11 - (d3);
                    float dd2 = d2 - (d10);
                    float dd3 = d7 + (d11);
                    float dd4 = d6 + (d10);
                    float dd5 = d7 - (d11);
                    float dd6 = d10 - (d6);
                    float dd7 = d15 - (d7);
                    float dd8 = d6 - (d14);

                    float v1 = d1 - d9 + dd1;
                    float v2 = dd2 - dd1;  //
                    float v3 = -dd1 - dd2; //
                    float v4 = dd2 - d4 + d12;

                    float v5 = d5 + d9 - dd3;
                    float v6 = dd4 + dd3;
                    float v7 = dd3 - dd4;
                    float v8 = dd4 - d8 - d12;

                    float v9 = d9 - d5 + dd5;
                    float v10 = dd6 - dd5;
                    float v11 = -(dd6 + dd5);
                    float v12 = dd6 + d8 - d12;

                    float v13 = d5 - d13 + dd7;
                    float v14 = dd8 - dd7;
                    float v15 = -dd7 - dd8;
                    float v16 = dd8 - d8 + d16;

                    // U . V
                    float m1 = v1 * U[u_idx];
                    float m2 = v2 * U[u_idx + 1];
                    float m3 = v3 * U[u_idx + 2];
                    float m4 = v4 * U[u_idx + 3];
                    float m5 = v5 * U[u_idx + 4];
                    float m6 = v6 * U[u_idx + 5];
                    float m7 = v7 * U[u_idx + 6];
                    float m8 = v8 * U[u_idx + 7];
                    float m9 = v9 * U[u_idx + 8];
                    float m10 = v10 * U[u_idx + 9];
                    float m11 = v11 * U[u_idx + 10];
                    float m12 = v12 * U[u_idx + 11];
                    float m13 = v13 * U[u_idx + 12];
                    float m14 = v14 * U[u_idx + 13];
                    float m15 = v15 * U[u_idx + 14];
                    float m16 = v16 * U[u_idx + 15];

                    // 4. output transfom
                    float sub_y1 = m2 + m6 + m10;
                    float sub_y2 = m3 + m7 + m11;
                    float sub_y3 = m6 - m10 - m14;
                    float sub_y4 = m7 - m11 - m15;

                    y1 += m1 + m5 + m9 + sub_y1 + sub_y2;
                    y2 += sub_y1 - sub_y2 - m4 - m8 - m12;
                    y3 += m5 - m9 - m13 + sub_y3 + sub_y4;
                    y4 += sub_y3 - sub_y4 - m8 + m12 + m16;
                }
                dst[ot_idx1] = y1;
                dst[ot_idx1 + 1] = y2;
                dst[ot_idx3] = y3;
                dst[ot_idx3 + 1] = y4;
            }
        }
    }
    free(U);
}

void zeroPadding(float* output,float* input, int input_n, int input_c, int input_h, int input_w, int pad_l, int pad_r, int pad_t, int pad_b)
{

	int temp1 = input_w * input_h * input_c;
	int temp1o = (input_h + pad_t + pad_b) * (input_w + pad_l + pad_r) * input_c;
	for (int n_idx = 0;n_idx < input_n; n_idx++)
	{
		int temp2 = n_idx * temp1;
		int temp2o = n_idx * temp1o;
		for (int c_idx = 0; c_idx < input_c;c_idx++)
		{
			int temp3 = c_idx * input_w * input_h + temp2;
			int temp3o = c_idx * (input_w + pad_l + pad_r) * (input_h + pad_t + pad_b) + temp2o;
			for (int h_idx = 0; h_idx < input_h; h_idx++)
			{
				int temp4 = h_idx * input_w + temp3;
				int temp4o = (h_idx + pad_t) * (input_w + pad_l + pad_r) + pad_l + temp3o;

				for (int w_idx = 0; w_idx < input_w; w_idx++)
				{
					int g_idx = w_idx + temp4;
					int g_idx_Output = w_idx + temp4o;
					output[g_idx_Output] = input[g_idx];
				}
			}
		}
	}
}

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct test_model {
    struct ggml_tensor * image;
    struct ggml_tensor * kernel_w;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

void load_model(test_model & model, bool use_gpu = false) {
    // create data
    const int KW = 3, KH = 3, IC = 1, OC = 1;
    const int IW = 8, IH = 8, N = 1;

    // Initialize kernel data
    const int kernel_elements = KW * KH * IC * OC;
    float kernel_d[kernel_elements] = {
        2.f, 1.0f, 4.f,
        2.5f, 1.5f, 2.f,
        2.f, 1.5f, 4.f,
    };

    // Convert adata to fp16 format
    std::vector<ggml_fp16_t> kernel_d_f16(kernel_elements);
    ggml_fp32_to_fp16_row(kernel_d, kernel_d_f16.data(), kernel_elements);

    // Initialize image data
    const int image_elements = IW * IH * IC * N;
    float image_d[image_elements] = {
        2.5f, 3.f, 4.f, 1.f, 2.f, 2.f, 2.f, 2.f,
        2.5f, 3.f, 4.f, 1.f, 2.f, 2.f, 2.f, 3.f,
        2.5f, 3.f, 4.f, 1.f, 2.f, 2.f, 1.f, 4.f,
        2.5f, 3.f, 1.f, 1.f, 2.f, 2.f, 3.f, 2.f,
        2.5f, 3.f, 4.f, 3.f, 1.f, 2.f, 2.f, 2.5f,
        2.5f, 3.f, 4.f, 1.f, 2.f, 2.f, 2.f, 2.f,
        2.5f, 3.f, 1.f, 4.f, 2.f, 2.f, 2.f, 2.f,
        2.5f, 3.f, 4.f, 1.f, 2.f, 2.f, 1.f, 2.f
    };

    size_t buffer_size = 0;
    {
        buffer_size += kernel_elements * ggml_type_sizef(GGML_TYPE_F16); // tensor a
        buffer_size += image_elements * ggml_type_sizef(GGML_TYPE_F32); // tensor b
        buffer_size += 1024; // overhead
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %0.2f KB\n", __func__, (buffer_size / 1024.f));

    int num_tensors = 2;
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    // initialize the backend
#ifdef GGML_USE_CUBLAS
    if (use_gpu) {
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init(0);
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (use_gpu) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        ggml_metal_log_set_callback(ggml_log_callback_default, nullptr);
        model.backend = ggml_backend_metal_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#endif

    if(!model.backend) {
        // fallback to CPU backend
        model.backend = ggml_backend_cpu_init();
    }

    model.buffer = ggml_backend_alloc_buffer(model.backend, buffer_size);

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.kernel_w = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16,  KW, KH, IC, OC);
    model.image = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32, IW, IH, IC, N);

    // create a allocator
    ggml_allocr * alloc = ggml_allocr_new_from_buffer(model.buffer);

    // alloc memory
    ggml_allocr_alloc(alloc, model.kernel_w);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.kernel_w->data, kernel_d_f16.data(), ggml_nbytes(model.kernel_w));
    } else {
        ggml_backend_tensor_set(model.kernel_w, kernel_d_f16.data(), 0, ggml_nbytes(model.kernel_w));
    }

    // alloc memory
    ggml_allocr_alloc(alloc, model.image);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.image->data, image_d, ggml_nbytes(model.image));
    } else {
        ggml_backend_tensor_set(model.image, image_d, 0, ggml_nbytes(model.image));
    }

    ggml_allocr_free(alloc);

    // Winograd
    int pad = 1;
    int OW = ((IW + 2*pad - KW) / 1) + 1;
    int OH = ((IH + 2*pad - KH) / 1) + 1;
    float * wino_conv2d = new float[OW * OH * OC];
    int image_pad_sz = (IW + 2*pad) * (IH + 2*pad) * IC;
    float * image_padded = new float[image_pad_sz];
    printf("mem size: %.2f KB", (image_pad_sz * sizeof(float) + OW*OH*OC*sizeof(float) + 16*sizeof(float)) / 1024.f);
    memset(image_padded, 0, image_pad_sz*sizeof(float));
    zeroPadding(image_padded, image_d, 1, IC, IH, IW, pad, pad, pad, pad);
    winogradConv2d(wino_conv2d, image_padded, kernel_d, IC, IH + 2*pad, IW + 2*pad, OC);
    printf("\nConvolution with Winograd\n");
    for(int i = 0; i < (OW * OH * OC);i++) {
        if(i> 0 && i % OW == 0) {
            printf("\n");
        }
        printf("%.3ff, ", wino_conv2d[i]);
    }
    printf("\n");
}

struct ggml_cgraph * build_graph(const test_model& model, struct ggml_allocr * allocr) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);

    int s0 = 1;
    int s1 = 1;
    int p0 = 1;
    int p1 = 1;
    int d0 = 1;
    int d1 = 1;

    // split conv2d in fundamental methods for test unit
    struct ggml_tensor* im2col_0 = ggml_im2col(ctx0, model.kernel_w, model.image, s0, s1, p0, p1, d0, d1, true);
    ggml_set_name(im2col_0, "im2col_res");
    ggml_build_forward_expand(gf, im2col_0);

    // recalculate for avoid fragmentation
    struct ggml_tensor* conv2d_res = ggml_conv_2d(ctx0, model.kernel_w, model.image, s0, s1, p0, p1, d0, d1);
    ggml_set_name(conv2d_res, "conv2d_res");
    ggml_build_forward_expand(gf, conv2d_res);

    ggml_free(ctx0);
    return gf;
}

struct ggml_cgraph * compute_graph(const test_model & model, struct ggml_allocr * allocr) {
    // reset the allocator to free all the memory allocated during the previous inference
    ggml_allocr_reset(allocr);

    struct ggml_cgraph * gf = build_graph(model, allocr);

    // allocate tensors
    ggml_allocr_alloc_graph(allocr, gf);
    int n_threads = 1;

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend)) {
        ggml_backend_metal_set_n_cb(model.backend, n_threads);
    }
#endif

    ggml_backend_graph_compute(model.backend, gf);

    //ggml_graph_print(gf);

    return gf;
}

int main(void)
{
    ggml_time_init();

    test_model model;
    load_model(model, true);

    ggml_backend_buffer_t buf_compute; // for compute
    struct ggml_allocr * allocr = NULL;

    {
        allocr = ggml_allocr_new_measure_from_backend(model.backend);

        //create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = build_graph(model, allocr);
        size_t mem_size = ggml_allocr_alloc_graph(allocr, gf);
        ggml_allocr_free(allocr);

        // compute the required memory
        buf_compute = ggml_backend_alloc_buffer(model.backend, mem_size);
        allocr = ggml_allocr_new_from_buffer(buf_compute);
        fprintf(stderr, "%s: compute buffer size: %.2f KB\n", __func__, mem_size / 1024.0f);
    }

    struct ggml_cgraph * gf_res = compute_graph(model, allocr);

    struct ggml_tensor * im2col_res = NULL;
    struct ggml_tensor * conv2d_res = NULL;

    for(int i = 0; i < gf_res->n_nodes; i++) {
        if(strcmp(ggml_get_name(gf_res->nodes[i]), "im2col_res") == 0) {
            im2col_res = gf_res->nodes[i];
        } else if(strcmp(ggml_get_name(gf_res->nodes[i]), "conv2d_res") == 0) {
            conv2d_res = gf_res->nodes[i];
        }
    }

    uint16_t* im2col_data = new uint16_t[ggml_nelements(im2col_res)];
    float* conv2d_data = new float[ggml_nelements(conv2d_res)];

    ggml_backend_tensor_get(im2col_res, im2col_data, 0, ggml_nbytes(im2col_res));
    ggml_backend_tensor_get(conv2d_res, conv2d_data, 0, ggml_nbytes(conv2d_res));

    const int n_conv2d_test = 64;
    const int n_im2col_test = 576;

    float expected_conv2d [n_conv2d_test] = {
        25.500f, 44.250f, 31.500f, 33.000f, 22.500f, 27.000f, 31.000f, 16.500f,
        40.000f, 68.250f, 45.500f, 50.000f, 34.500f, 37.000f, 49.500f, 23.500f,
        40.000f, 56.250f, 41.000f, 44.000f, 34.500f, 43.000f, 49.000f, 24.500f,
        40.000f, 62.250f, 49.000f, 41.500f, 37.000f, 37.000f, 51.500f, 24.250f,
        40.000f, 56.250f, 46.500f, 45.000f, 38.000f, 42.500f, 43.000f, 23.750f,
        40.000f, 56.250f, 61.000f, 46.500f, 43.500f, 39.000f, 43.000f, 21.500f,
        40.000f, 62.250f, 47.000f, 47.000f, 42.000f, 37.000f, 39.500f, 19.000f,
        24.250f, 30.750f, 38.500f, 29.500f, 27.500f, 24.000f, 24.500f, 11.500f };

    uint16_t expected_im2col[n_im2col_test] = {
        0, 0, 0, 0, 16640, 16896, 0, 16640, 16896,
        0, 0, 0, 16640, 16896, 17408, 16640, 16896, 17408,
        0, 0, 0, 16896, 17408, 15360, 16896, 17408, 15360,
        0, 0, 0, 17408, 15360, 16384, 17408, 15360, 16384,
        0, 0, 0, 15360, 16384, 16384, 15360, 16384, 16384,
        0, 0, 0, 16384, 16384, 16384, 16384, 16384, 16384,
        0, 0, 0, 16384, 16384, 16384, 16384, 16384, 16896,
        0, 0, 0, 16384, 16384, 0, 16384, 16896, 0,
        0, 16640, 16896, 0, 16640, 16896, 0, 16640, 16896,
        16640, 16896, 17408, 16640, 16896, 17408, 16640, 16896, 17408,
        16896, 17408, 15360, 16896, 17408, 15360, 16896, 17408, 15360,
        17408, 15360, 16384, 17408, 15360, 16384, 17408, 15360, 16384,
        15360, 16384, 16384, 15360, 16384, 16384, 15360, 16384, 16384,
        16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 15360,
        16384, 16384, 16384, 16384, 16384, 16896, 16384, 15360, 17408,
        16384, 16384, 0, 16384, 16896, 0, 15360, 17408, 0,
        0, 16640, 16896, 0, 16640, 16896, 0, 16640, 16896,
        16640, 16896, 17408, 16640, 16896, 17408, 16640, 16896, 15360,
        16896, 17408, 15360, 16896, 17408, 15360, 16896, 15360, 15360,
        17408, 15360, 16384, 17408, 15360, 16384, 15360, 15360, 16384,
        15360, 16384, 16384, 15360, 16384, 16384, 15360, 16384, 16384,
        16384, 16384, 16384, 16384, 16384, 15360, 16384, 16384, 16896,
        16384, 16384, 16896, 16384, 15360, 17408, 16384, 16896, 16384,
        16384, 16896, 0, 15360, 17408, 0, 16896, 16384, 0,
        0, 16640, 16896, 0, 16640, 16896, 0, 16640, 16896,
        16640, 16896, 17408, 16640, 16896, 15360, 16640, 16896, 17408,
        16896, 17408, 15360, 16896, 15360, 15360, 16896, 17408, 16896,
        17408, 15360, 16384, 15360, 15360, 16384, 17408, 16896, 15360,
        15360, 16384, 16384, 15360, 16384, 16384, 16896, 15360, 16384,
        16384, 16384, 15360, 16384, 16384, 16896, 15360, 16384, 16384,
        16384, 15360, 17408, 16384, 16896, 16384, 16384, 16384, 16640,
        15360, 17408, 0, 16896, 16384, 0, 16384, 16640, 0,
        0, 16640, 16896, 0, 16640, 16896, 0, 16640, 16896,
        16640, 16896, 15360, 16640, 16896, 17408, 16640, 16896, 17408,
        16896, 15360, 15360, 16896, 17408, 16896, 16896, 17408, 15360,
        15360, 15360, 16384, 17408, 16896, 15360, 17408, 15360, 16384,
        15360, 16384, 16384, 16896, 15360, 16384, 15360, 16384, 16384,
        16384, 16384, 16896, 15360, 16384, 16384, 16384, 16384, 16384,
        16384, 16896, 16384, 16384, 16384, 16640, 16384, 16384, 16384,
        16896, 16384, 0, 16384, 16640, 0, 16384, 16384, 0,
        0, 16640, 16896, 0, 16640, 16896, 0, 16640, 16896,
        16640, 16896, 17408, 16640, 16896, 17408, 16640, 16896, 15360,
        16896, 17408, 16896, 16896, 17408, 15360, 16896, 15360, 17408,
        17408, 16896, 15360, 17408, 15360, 16384, 15360, 17408, 16384,
        16896, 15360, 16384, 15360, 16384, 16384, 17408, 16384, 16384,
        15360, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384,
        16384, 16384, 16640, 16384, 16384, 16384, 16384, 16384, 16384,
        16384, 16640, 0, 16384, 16384, 0, 16384, 16384, 0,
        0, 16640, 16896, 0, 16640, 16896, 0, 16640, 16896,
        16640, 16896, 17408, 16640, 16896, 15360, 16640, 16896, 17408,
        16896, 17408, 15360, 16896, 15360, 17408, 16896, 17408, 15360,
        17408, 15360, 16384, 15360, 17408, 16384, 17408, 15360, 16384,
        15360, 16384, 16384, 17408, 16384, 16384, 15360, 16384, 16384,
        16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 15360,
        16384, 16384, 16384, 16384, 16384, 16384, 16384, 15360, 16384,
        16384, 16384, 0, 16384, 16384, 0, 15360, 16384, 0,
        0, 16640, 16896, 0, 16640, 16896, 0, 0, 0,
        16640, 16896, 15360, 16640, 16896, 17408, 0, 0, 0,
        16896, 15360, 17408, 16896, 17408, 15360, 0, 0, 0,
        15360, 17408, 16384, 17408, 15360, 16384, 0, 0, 0,
        17408, 16384, 16384, 15360, 16384, 16384, 0, 0, 0,
        16384, 16384, 16384, 16384, 16384, 15360, 0, 0, 0,
        16384, 16384, 16384, 16384, 15360, 16384, 0, 0, 0,
        16384, 16384, 0, 15360, 16384, 0, 0, 0, 0
    };

    printf("\nConvolution with im2col\n");

    for(int i = 0; i < ggml_nelements(conv2d_res);i++) {
        if(i> 0 && i % conv2d_res->ne[0] == 0) {
            printf("\n");
        }
        printf("%.3ff, ", conv2d_data[i]);
    }

    printf("\nPerforming test:\n");

    bool passed = true;
    for(int i = 0; i < n_im2col_test; i++) {
        if(
            im2col_data[i] != expected_im2col[i]) {
            passed = false;
            break;
        }
    }

    printf("ggml_im2col (%d): %s\n", (int) ggml_nelements(im2col_res), passed && (ggml_nelements(im2col_res) == n_im2col_test) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");

    passed = true;
    for(int i = 0; i < n_conv2d_test; i++) {
        if(conv2d_data[i] != expected_conv2d[i]) {
            passed = false;
            break;
        }
    }

    printf("ggml_conv2d (%d): %s\n", (int) ggml_nelements(conv2d_res), passed && (ggml_nelements(conv2d_res) == n_conv2d_test) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");

    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_buffer_free(buf_compute);
    ggml_backend_free(model.backend);
    return 0;
}
