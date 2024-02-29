#include "ggml/ggml.h"
#include "superpoint-image.h"
#include "utils.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <functional>
#include <numeric>


#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif



int clip(int val, int max)
{
  if (val < 0)
    return 0;
  return std::min(val, max - 1);
}


static std::vector<float> softmax(const std::vector<float> & logits) {
    std::vector<float> probs(logits.size());
    float max_logit = logits[0];
    for (float v : logits) max_logit = std::max(max_logit, v);
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        // Subtract the maximum logit value from the current logit value for numerical stability
        const float logit = logits[i] - max_logit;
        const float exp_logit = expf(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (size_t i = 0; i < probs.size(); i++) probs[i] /= sum_exp;
    return probs;
}

static struct ggml_tensor* brute_permute(ggml_context * ctx, struct ggml_tensor *input, int d0, int d1, int d2, int d3)
{
    //assert it is contigous
    //STEP 1: get each stride at src tensor
    int dims[4];
    int strides[4];
    dims[d0] = input->ne[0];
    dims[d1] = input->ne[1];
    dims[d2] = input->ne[2];
    dims[d3] = input->ne[3];

    //STEP 2: based on the permute result, recalcute the stride
    //get element_size
    strides[d0] = input->nb[0]/sizeof(float);
    strides[d1] = input->nb[1]/sizeof(float);
    strides[d2] = input->nb[2]/sizeof(float);
    strides[d3] = input->nb[3]/sizeof(float);
    //create a new tensor with the wanted shape
    auto new_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                    dims[0], dims[1], dims[2], dims[3]);

    float* data_src = ggml_get_data_f32(input);
    float* data_dst = ggml_get_data_f32(new_tensor);
    //do the permutation
    //use for loop, to reallocate the data
    int cnt = 0;
    for (int h =0; h <dims[3]; h++)
        for (int i =0; i < dims[2]; i++)
            for (int j =0; j <dims[1]; j++)
                for (int k =0; k<dims[0]; k++)//stride of d0
                {
                    int index = k * strides[0] + j * strides[1] + i * strides[2] + h * strides[3];
                    // printf("index %d\n", index);
                    data_dst[cnt] = data_src[index];
                    cnt++;
                }
    return new_tensor;
}



static struct ggml_tensor* softmax_semi(ggml_context * ctx, struct ggml_tensor *input)
{
    int w = input->ne[0];
    int h = input->ne[1];
    int c = input->ne[2];
    int num = w * h *c;


    printf("before softmax Shape:  %3d x %3d x %4d x %3d\n", w, h, c, (int)input->ne[3]);
    //well, it seems like channel is at the 3rd channel, to iterate over it, shift it to the first dim
#define NMS_V1
#ifdef NMS_V1
    auto _tensor = brute_permute(ctx, input,2,1,0,3);
#else
    auto _tensor = brute_permute(ctx, input,1,2,0,3);

#endif
    int ne0 = _tensor->ne[0];
    int ne1 = _tensor->ne[1];
    int ne2 = _tensor->ne[2];
    int ne3 = _tensor->ne[3];
    int tensor_size = ne0 * ne1 * ne2 * ne3;

    struct ggml_tensor* nodust = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, _tensor->ne[1], _tensor->ne[2], (int)_tensor->ne[3]);
    int channel  = 65;//c

    for(int i =0; i < num/channel; i+=1)
    {
        int in_offset = i * channel;
        int out_offset  = i * (channel -1);
        float* in_arr = ggml_get_data_f32(_tensor)+ in_offset;
        float* out_arr = ggml_get_data_f32(nodust)+ out_offset;
        std::vector<float> in_vec(in_arr, in_arr + channel);
        std::vector<float>  vec = softmax(in_vec);
        std::copy(vec.begin(), vec.end() - 1, out_arr);
    }
    //TODO: the operation above seems not usual?

    printf("Final Shape:  %3lld x %3lld x %4lld x %3d\n", nodust->ne[0], nodust->ne[1], nodust->ne[2], (int)nodust->ne[3]);
    return nodust;
}

/*
A interesting bug, you have to flatten the tensor(view it as 1d) to make permutation work...
*/
static struct ggml_tensor* heatmap_semi(ggml_context * ctx, struct ggml_tensor *scores)
{
    int cell =8;
    int Hc = scores->ne[2];
    int Wc = scores->ne[1];

    scores = ggml_reshape_4d(ctx, scores,  8, 8, Wc, Hc);//h,w,8,8

    auto heatmap = brute_permute(ctx, scores, 0,2,1,3);

    print_shape(100, heatmap);
    // write_array("heatmap.txt", heatmap);


    return heatmap;

}

void screen_keypoints(struct ggml_tensor* tensor,
                      int H,
                      int W,
                      int cell,
                      std::vector<PointT>& pts /* OUTPUT */)
{
    float * data =  ggml_get_data_f32(tensor);

    const int HO = H/cell;
    const int WO = W/cell;
    const int border_remove = 10;
    const float conf_thresh = 0.015;
    int output_semi_dim0 = cell * cell;
    int output_semi_dim1 = HO;
    int output_semi_dim2 = WO;


    const int left_margin = border_remove;
    const int right_margin = W - border_remove;
    const int top_margin = border_remove;
    const int bottom_margin = H - border_remove;

    auto get_coord = [H, W](int a, int b, int c, int &row, int &col) -> bool {
        row = (b << 3) + (a >> 3);
        col = (c << 3) + (a bitand 0x07);
        return ((row >= 0) && (row < H) && (col >= 0) && (col < W));
    };
    auto get_offset = [HO, WO](int a, int b, int c) -> int {
        return ( (a * HO * WO) + (b * WO) + c );
    };
    auto action_on_each = [&](int k, int i, int j) {
        float prob = data[get_offset(k, i, j)];
        if (prob > conf_thresh)
        {
            int row = -1, col = -1;
            get_coord(k, i, j, row, col);

            if ( row < top_margin || row >= bottom_margin ||
                 col < left_margin  || col >= right_margin )
            {
                ; /* drop point along border */
            }
            else
            {
                /* qualified points */
                pts.emplace_back( col /* x */, row /* y */, prob );
            }
        }
    };

    pts.clear();
    pts.reserve(4000);

    /* measurement util */
    // Timer t;

    for (int k = 0; k < output_semi_dim0; k++ )
    {
        for (int i = 0; i < output_semi_dim1; i++)
        {
            for (int j = 0; j < output_semi_dim2; j++)
            {
                action_on_each(k, i, j);
            }
        }
    }

}


void nms_fast(std::vector<PointT>& corners, int H, int W, int keypoints_num, int nms_dist)
{
    if ( corners.empty() ) return;

    std::sort( corners.begin(), corners.end(),
               [](const PointT& a, const PointT& b) -> bool {
                  return (a.conf > b.conf); }
             );

    const int pad = nms_dist;
    const int grid_H = H + 2 * pad;
    const int grid_W = W + 2 * pad;
    std::vector<std::vector<float>> grid(grid_H, std::vector<float>(grid_W, 0));


    for (auto & pt : corners)
    {
        const int row = pad + pt.y;
        const int col = pad + pt.x;
        grid[row][col] = 1;
    }

    /* reserve survivors */
    std::vector<PointT> survivors;
    survivors.reserve(1000);

    for (auto & pt : corners)
    {
        const int row = pad + pt.y;
        const int col = pad + pt.x;

        if (grid[row][col] == 1)
        {
            for(int c = col - pad; c< std::min(col + pad, grid_W); c++)
                for (int r = row - pad; r< std::min(row + pad, grid_H); r++)
                {
                    grid[r][c] = 0;
                }
            /* keep the corner */
            grid[row][col] == -1;
            survivors.push_back(pt);
        }
    }

    corners.clear();
    corners = std::move(survivors);

    while (corners.size() > (size_t)keypoints_num)
    {
        corners.pop_back();
    }
}


static void postprocess_semi(ggml_context * ctx, superpoint_image & img, struct ggml_tensor *input, std::vector<PointT>& pts)
{
    auto scores = softmax_semi(ctx, input);
    printf("screen keypoints\n");
    //TODO: improve efficiency of permutation
    scores  = brute_permute(ctx, scores, 2,1,0,3);
    // brute_permute()
    screen_keypoints( scores, img.h, img.w, 8, pts);
    nms_fast(pts, img.h, img.w, 4000, 4);
    // printf("points size %d\n", pts.size());
    // write_points("points.txt", pts);

}

void normalize_keypoints(const std::vector<PointT> &keypoints, std::vector<PointT> &keypoints_norm,
                         int h, int w)
{
  for (auto &kp : keypoints)
  {
    PointT keypoint;
    keypoint.conf = kp.conf;

    keypoint.x = (float)kp.x / (0.5 *w) - 1;
    keypoint.y = (float)kp.y / (0.5 *h) - 1;
    keypoints_norm.push_back(keypoint);
    // std::cout<<"keypoint: "<<keypoint.x<<", "<<keypoint.y<<std::endl;
  }
//   exit(0);
}

void grid_sample(const float *input, std::vector<PointT> &grid,
                 std::vector<std::vector<double>> &output, int dim, int h, int w)
{
  for (auto &g : grid)
  {
    double ix = ((g.x + 1) / 2) * (w - 1);
    double iy = ((g.y + 1) / 2) * (h - 1);
    // std::cout<<"ix: "<<ix<<", iy: "<<iy<<std::endl;

    int ix_nw = clip(std::floor(ix), w);
    int iy_nw = clip(std::floor(iy), h);

    int ix_ne = clip(ix_nw + 1, w);
    int iy_ne = clip(iy_nw, h);

    int ix_sw = clip(ix_nw, w);
    int iy_sw = clip(iy_nw + 1, h);

    int ix_se = clip(ix_nw + 1, w);
    int iy_se = clip(iy_nw + 1, h);

    double nw = (ix_se - ix) * (iy_se - iy);
    double ne = (ix - ix_sw) * (iy_sw - iy);
    double sw = (ix_ne - ix) * (iy - iy_ne);
    double se = (ix - ix_nw) * (iy - iy_nw);

    std::vector<double> descriptor;
    for (int i = 0; i < dim; ++i)
    {
      // 256x60x106 dhw
      //TODO: check this index
      // x * height * depth + y * depth + z
      float nw_val = input[i * h * w + iy_nw * w + ix_nw];
      float ne_val = input[i * h * w + iy_ne * w + ix_ne];
      float sw_val = input[i * h * w + iy_sw * w + ix_sw];
      float se_val = input[i * h * w + iy_se * w + ix_se];
      descriptor.push_back(nw_val * nw + ne_val * ne + sw_val * sw + se_val * se);
    }
    output.push_back(descriptor);
  }
    // exit(0);
}




template <typename Iter_T>
double vector_normalize(Iter_T first, Iter_T last)
{
  return sqrt(std::inner_product(first, last, first, 0.0));
}

void normalize_descriptors(std::vector<std::vector<double>> &dest_descriptors)
{
  for (auto &descriptor : dest_descriptors)
  {
    double norm_inv = 1.0 / vector_normalize(descriptor.begin(), descriptor.end());
    std::transform(descriptor.begin(), descriptor.end(), descriptor.begin(),
                   std::bind1st(std::multiplies<double>(), norm_inv));
  }
}

static void postprocess_desc(ggml_context * ctx,
                             superpoint_image & img,
                             struct ggml_tensor *desc,
                             std::vector<PointT>& pts,
                             std::vector<std::vector<double>>& descriptors)
{
    std::vector<PointT> keypoints_norm;
    int h = img.h;
    int w = img.w;
    normalize_keypoints(pts, keypoints_norm, h, w);
    float* desc_ptr =  ggml_get_data_f32(desc);
    int dim = 256;
    grid_sample(desc_ptr, keypoints_norm, descriptors, dim, h/8., w/8.);
    normalize_descriptors(descriptors);
}


struct conv2d_layer {
    struct ggml_tensor * weights;
    struct ggml_tensor * biases;
    // struct ggml_tensor * scales;
    // struct ggml_tensor * rolling_mean;
    // struct ggml_tensor * rolling_variance;
    int padding = 1;
    bool batch_normalize = false;
    bool activate = true; // true for relu, false for linear
};

struct superpoint_model
{
    int width = 640;
    int height = 480;
    std::vector<conv2d_layer> conv2d_layers;
    struct ggml_context * ctx;
};

static bool load_model(const std::string & fname, superpoint_model& model) {
    struct gguf_init_params params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &model.ctx,
    };
    gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);
    if (!ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }
    model.width  = 640;
    model.height = 480;
    //TODO: switch the size 13 too!
    model.conv2d_layers.resize(12);

    for (int i = 0; i < (int)model.conv2d_layers.size(); i++) {
        char name[256];
        snprintf(name, sizeof(name), "l%d_weights", i);
        //ggml_fp32_to_fp16
        model.conv2d_layers[i].weights = ggml_get_tensor(model.ctx, name);
        //The weights are loaded as fp16
        model.conv2d_layers[i].weights->type = ggml_type::GGML_TYPE_F16;
        snprintf(name, sizeof(name), "l%d_biases", i);
        model.conv2d_layers[i].biases = ggml_get_tensor(model.ctx, name);
    }

    //layers without relu
    model.conv2d_layers[9].activate = false;
    model.conv2d_layers[9].padding = 0;
    model.conv2d_layers[11].activate = false;
    model.conv2d_layers[11].padding = 0;





    return true;
}

static ggml_tensor * apply_conv2d(ggml_context * ctx, ggml_tensor * input, const conv2d_layer & layer)
{
    // struct ggml_tensor * result = ggml_conv_1d(ctx, layer.weights, input, 1, 1, 1);
    struct ggml_tensor * result = ggml_conv_2d(ctx, layer.weights, input, 1, 1, layer.padding, layer.padding, 1, 1);

    result = ggml_add(ctx, result, ggml_repeat(ctx, layer.biases, result));
    if (layer.activate) {
        //implement normal relu
        result = ggml_relu(ctx, result);
    }
    return result;
}

static void activate_array(float * x, const int n)
{
    // logistic activation
    for (int i = 0; i < n; i++) {
        x[i] = 1./(1. + exp(-x[i]));
    }
}


static float get_color(int c, int x, int max)
{
    float colors[6][3] = { {1,0,1}, {0,0,1}, {0,1,1}, {0,1,0}, {1,1,0}, {1,0,0} };
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    return r;
}



void inference(superpoint_image & img,
               const superpoint_model & model,
               float thresh,
               std::vector<PointT>& pts,
               std::vector<std::vector<double>>& descriptors)

{
    //TODO: modifiy the size, bc it is too large
    static size_t buf_size = 20000000 * sizeof(float) * 40;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx0 =  ggml_init(params);
    // model.ctx = ctx0;
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    // std::vector<detection> detections;
    //reshape the image
    superpoint_image sized = letterbox_image(img, model.width, model.height);

    //allovate datasize
    struct ggml_tensor * input = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, model.width, model.height, 1, 1);
    std::memcpy(input->data, img.data.data(), ggml_nbytes(input));
    ggml_set_name(input, "input");
    print_shape(0, input);

    //x = self.relu(self.conv1a(x))
    struct ggml_tensor * result = apply_conv2d(ctx0, input, model.conv2d_layers[0]);
    result = apply_conv2d(ctx0, result, model.conv2d_layers[1]);
    result = ggml_pool_2d(ctx0, result, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    print_shape(2, result);
    //x = self.relu(self.conv2a(x))
    result = apply_conv2d(ctx0, result, model.conv2d_layers[2]);
    print_shape(3, result);
    //x = self.relu(self.conv2b(x))
    result = apply_conv2d(ctx0, result, model.conv2d_layers[3]);
    print_shape(4, result);
    //x = self.pool(x)
    result = ggml_pool_2d(ctx0, result, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    print_shape(5, result);
    //x = self.relu(self.conv3a(x))
    result = apply_conv2d(ctx0, result, model.conv2d_layers[4]);
    // for further connections
    // struct ggml_tensor * layer_8 = result;
    print_shape(6, result);
    // x = self.relu(self.conv3b(x))
    result = apply_conv2d(ctx0, result, model.conv2d_layers[5]);
    print_shape(7, result);
    //    x = self.pool(x)
    result = ggml_pool_2d(ctx0, result, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    print_shape(8, result);
    result = apply_conv2d(ctx0, result, model.conv2d_layers[6]);
    print_shape(9, result);
    // x = self.relu(self.conv4b(x))
    struct ggml_tensor * encoder_output = apply_conv2d(ctx0, result, model.conv2d_layers[7]);

    result = apply_conv2d(ctx0, encoder_output, model.conv2d_layers[8]);
    print_shape(11, result);
    struct ggml_tensor * semi = apply_conv2d(ctx0, result, model.conv2d_layers[9]);
    print_shape(12, semi);

    result = apply_conv2d(ctx0, encoder_output, model.conv2d_layers[10]);
    print_shape(13, result);

    struct ggml_tensor * desc = apply_conv2d(ctx0, result, model.conv2d_layers[11]);
    print_shape(14, desc);
    ggml_build_forward_expand(gf, semi);
    ggml_build_forward_expand(gf, desc);
    ggml_graph_compute_with_ctx(ctx0, gf, 1);
    const int64_t t_start_ms = ggml_time_ms();
    postprocess_semi(ctx0, sized, semi, pts);
    postprocess_desc(ctx0, sized, desc, pts, descriptors);
    const int64_t t_detect_ms = ggml_time_ms() - t_start_ms;
    printf("superpoint  postprocessing time: %f sec.)\n", t_detect_ms / 1000.0f);




}

struct superpoint_params {
    float thresh          = 0.5;
    std::string model     = "superpoint.gguf";
    std::string fname_inp = "dog_color.jpg";
    std::string fname_out = "result.jpg";
};

void superpoint_print_usage(int argc, char ** argv, const superpoint_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -th T, --thresh T     detection threshold (default: %.2f)\n", params.thresh);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  -i FNAME, --inp FNAME\n");
    fprintf(stderr, "                        input file (default: %s)\n", params.fname_inp.c_str());
    fprintf(stderr, "  -o FNAME, --out FNAME\n");
    fprintf(stderr, "                        output file (default: %s)\n", params.fname_out.c_str());
    fprintf(stderr, "\n");
}

bool superpoint_params_parse(int argc, char ** argv, superpoint_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-th" || arg == "--thresh") {
            params.thresh = std::stof(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-i" || arg == "--inp") {
            params.fname_inp = argv[++i];
        } else if (arg == "-o" || arg == "--out") {
            params.fname_out = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            superpoint_print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            superpoint_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

int main(int argc, char *argv[])
{
    ggml_time_init();
    superpoint_model model;

    superpoint_params params;
    if (!superpoint_params_parse(argc, argv, params)) {
        return 1;
    }
    if (!load_model(params.model, model)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
        return 1;
    }
    superpoint_image img(0,0,0);
    if (!load_image(params.fname_inp.c_str(), img, true)) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.fname_inp.c_str());
        return 1;
    }

    superpoint_image rgb_img(0,0,0);
    if (!load_image(params.fname_inp.c_str(), rgb_img, false)) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.fname_inp.c_str());
        return 1;
    }

    /*resize image to 640 480 currently is needed*/
    assert(img.w == 640);
    assert(img.h == 480);

    printf("start inference\n");
    std::vector<PointT> pts;
    std::vector<std::vector<double>>descriptors;
    const int64_t t_start_ms = ggml_time_ms();
    inference(img, model, params.thresh,pts, descriptors);
    const int64_t t_detect_ms = ggml_time_ms() - t_start_ms;
    printf("superpoint  inference time: %f sec.)\n", t_detect_ms / 1000.0f);

    //dump data
    write_points("points.txt", pts);
    write_descriptors("descs.txt", descriptors);

    //visualize points
    for(auto& pt:pts)
    {
        float red = get_color(2,0,5);
        float green = get_color(1,0,5);
        float blue = get_color(0,0,5);
        draw_point(rgb_img, pt.y, pt.x, red, green, blue);
    }
    if (!save_image(rgb_img, params.fname_out.c_str(), 80)) {
        fprintf(stderr, "%s: failed to save image to '%s'\n", __func__, params.fname_out.c_str());
        return 1;
    }
    ggml_free(model.ctx);

    return 0;
}
