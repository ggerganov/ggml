#include "conv-transpose-1d.cuh"

static  __global__ void conv_transpose_1d_kernel(
        const int s0, const int p0, const int d0,
        const int kernel_size, const int input_size, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const float * src0, const float * src1,  float * dst) {
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_index >= output_size) {
        return;
    }
    //printf("idx: %d stride %d\n", global_index,s0);

    int out_index = global_index / dst_ne0;

    for (int c = 0; c < src0_ne2; c++)
    {

        int idx = global_index % dst_ne0;

        int kernel_offset = (src0_ne0 * src0_ne1 * out_index) + (c * src0_ne0);
        int input_offset = src1_ne0 * c;

        if(global_index == 3 && s0 == 2)
        {
        printf("idx: %d ???: %d\n", global_index,src0_ne2);

        printf("idx: %d kernel offset: %d\n", global_index,kernel_offset);
        printf("idx: %d input offset: %d\n", global_index,input_offset);
        }

        int upper_bound = idx > src1_ne0-1 ? src1_ne0-1 : (int)(idx/s0)*s0; //inclusive
        /*
        int upper_bound = 0;
        while (upper_bound < idx){
            upper_bound +=1;
        }*/


        int lower_bound = idx - src0_ne0 + 1 >= 0 ? (int)(idx/s0)*s0 - src0_ne0 + 1 : 0;

        int initial_weight_idx = idx > src0_ne0 -1 ? src0_ne0-1 : idx;

        if(global_index == 3 && s0 == 2)
        {
        printf("idx: %d initial_weight_idx: %d\n", global_index,initial_weight_idx);
        printf("idx: %d upper bound: %d\n", global_index, upper_bound);
        printf("idx: %d lower bound: %d\n", global_index, lower_bound);
        }

        for (int i = 0; i < src1_ne0; i++)
        {
            if (!(idx >= i*s0 && idx < i*s0 + src0_ne0))
            {
                continue;
            }
            int weight_idx = idx - i*s0;


            if(global_index == 3 && s0 == 2)
            {
            //printf("idx: %d partial sum: %d x %d \n", global_index,src0[kernel_offset + (initial_weight_idx-(i-lower_bound))] , src1[input_offset+i]);
            //printf("idx: %d kernel_index: %d\n", global_index, kernel_offset + (initial_weight_idx-(i-lower_bound)));
            //printf("idx: %d input_index: %d\n", global_index, initial_weight_idx-(i-lower_bound));

            //printf("idx: %d input_index: %d\n", global_index, input_offset+i);

            }
            int test1 = src0[kernel_offset + weight_idx];
            int test2 =  src1[input_offset+i];
            if(global_index == 3 && s0 == 2)
            {
            //printf("idx: %d partial sum: %d x %d \n", global_index,src0[kernel_offset + (initial_weight_idx-(i-lower_bound))] , src1[input_offset+i]);
            //printf("idx: %d kernel_index: %d\n", global_index, kernel_offset + (initial_weight_idx-(i-lower_bound)));
            //printf("idx: %d input_index: %d\n", global_index, initial_weight_idx-(i-lower_bound));

            //printf("idx: %d input_index: %d\n", global_index, input_offset+i);
            printf("idx: %d test: %d x %d\n", global_index, test1, test2);

            }
            dst[global_index] += test1 * test2;
        }
        //dst[idx] = 7;
    }
}

static void conv_transpose_1d_f32_f32_cuda(
        const int s0, const int p0, const int d0,
        const int kernel_size, const int input_size, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const float * src0, const float * src1,  float * dst,
        cudaStream_t stream) {

    const int num_blocks = (output_size + CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE;
    conv_transpose_1d_kernel<<<num_blocks,CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE, 0, stream>>>(s0,p0,d0,kernel_size, input_size, output_size,
    src0_ne0, src0_ne1,  src0_ne2, src0_ne3,
    src1_ne0, src1_ne1,  src1_ne2, src1_ne3,
    dst_ne0,  dst_ne1,   dst_ne2,  dst_ne3,
  src0,src1, dst);
}

void ggml_cuda_op_conv_transpose_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;

    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *)dst->op_params;

    const int s0 = dst->op_params[0];
    const int p0 = 0;//opts[3];
    const int d0 = 1;//opts[4];

    const int64_t kernel_size = ggml_nelements(src0);
    const int64_t input_size = ggml_nelements(src1);
    const int64_t output_size =  ggml_nelements(dst);


    conv_transpose_1d_f32_f32_cuda( s0,p0,d0,kernel_size, input_size, output_size, 
    src0->ne[0],src0->ne[1],src0->ne[2],src0->ne[3],
    src1->ne[0],src1->ne[1],src1->ne[2],src1->ne[3],
    dst->ne[0],dst->ne[1],dst->ne[2],dst->ne[3],
    src0_d, src1_d, dst_d, stream);
}
