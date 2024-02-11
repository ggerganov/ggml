#define _USE_MATH_DEFINES // for M_PI
#define _CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnigns on Windows

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <thread>
#include <cinttypes>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// default hparams (ViT-B SAM)
struct sam_hparams {
    int32_t n_enc_state               = 768;
    int32_t n_enc_layer               = 12;
    int32_t n_enc_head                = 12;
    int32_t n_enc_out_chans           = 256;
    int32_t n_pt_embd                 = 4;
    int32_t n_dec_heads               = 8;
    int32_t ftype                     = 1;
    float   mask_threshold            = 0.f;
    float   iou_threshold             = 0.88f;
    float   stability_score_threshold = 0.95f;
    float   stability_score_offset    = 1.0f;
    float   eps                       = 1e-6f;
    float   eps_decoder_transformer   = 1e-5f;

    int32_t n_enc_head_dim() const { return n_enc_state / n_enc_head; }
    int32_t n_img_size()     const { return 1024; }
    int32_t n_window_size()  const { return 14; }
    int32_t n_patch_size()   const { return 16; }
    int32_t n_img_embd()     const { return n_img_size() / n_patch_size(); }

    std::vector<int32_t> global_attn_indices() const {
        switch (n_enc_state) {
            case  768: return {  2,  5,  8, 11 };
            case 1024: return {  5, 11, 17, 23 };
            case 1280: return {  7, 15, 23, 31 };
            default:
                {
                    fprintf(stderr, "%s: unsupported n_enc_state = %d\n", __func__, n_enc_state);
                } break;
        };

        return {};
    }

    bool is_global_attn(int32_t layer) const {
        const auto indices = global_attn_indices();

        for (const auto & idx : indices) {
            if (layer == idx) {
                return true;
            }
        }

        return false;
    }
};

struct sam_layer_enc {
    struct ggml_tensor * norm1_w;
    struct ggml_tensor * norm1_b;

    struct ggml_tensor * rel_pos_w;
    struct ggml_tensor * rel_pos_h;

    struct ggml_tensor * qkv_w;
    struct ggml_tensor * qkv_b;

    struct ggml_tensor * proj_w;
    struct ggml_tensor * proj_b;

    struct ggml_tensor * norm2_w;
    struct ggml_tensor * norm2_b;

    struct ggml_tensor * mlp_lin1_w;
    struct ggml_tensor * mlp_lin1_b;

    struct ggml_tensor * mlp_lin2_w;
    struct ggml_tensor * mlp_lin2_b;
};

struct sam_encoder_image {
    struct ggml_tensor * pe;

    struct ggml_tensor * proj_w;
    struct ggml_tensor * proj_b;

    struct ggml_tensor * neck_conv_0;
    struct ggml_tensor * neck_norm_0_w;
    struct ggml_tensor * neck_norm_0_b;
    struct ggml_tensor * neck_conv_1;
    struct ggml_tensor * neck_norm_1_w;
    struct ggml_tensor * neck_norm_1_b;

    std::vector<sam_layer_enc> layers;
};

struct sam_encoder_prompt {
    struct ggml_tensor * pe;

    struct ggml_tensor * not_a_pt_embd_w;
    std::vector<struct ggml_tensor *> pt_embd;

    struct ggml_tensor * no_mask_embd_w;
    //std::vector<struct ggml_tensor *> mask_down_w;
    //std::vector<struct ggml_tensor *> mask_down_b;
};

struct  sam_layer_dec_transformer_attn {
    // q_proj
    struct ggml_tensor * q_w;
    struct ggml_tensor * q_b;

    // k_proj
    struct ggml_tensor * k_w;
    struct ggml_tensor * k_b;

    // v_proj
    struct ggml_tensor * v_w;
    struct ggml_tensor * v_b;

    // out_proj
    struct ggml_tensor * out_w;
    struct ggml_tensor * out_b;
};

struct sam_layer_dec_transformer {
    sam_layer_dec_transformer_attn self_attn;

    // norm1
    struct ggml_tensor * norm1_w;
    struct ggml_tensor * norm1_b;

    sam_layer_dec_transformer_attn cross_attn_token_to_img;

    // norm2
    struct ggml_tensor * norm2_w;
    struct ggml_tensor * norm2_b;

    // mlp.lin1
    struct ggml_tensor * mlp_lin1_w;
    struct ggml_tensor * mlp_lin1_b;

    // mlp.lin2
    struct ggml_tensor * mlp_lin2_w;
    struct ggml_tensor * mlp_lin2_b;

    // norm3
    struct ggml_tensor * norm3_w;
    struct ggml_tensor * norm3_b;

    // norm4
    struct ggml_tensor * norm4_w;
    struct ggml_tensor * norm4_b;

    sam_layer_dec_transformer_attn cross_attn_img_to_token;
};

struct sam_layer_dec_output_hypernet_mlps {
    // mlps_*.layers.0
    struct ggml_tensor * w_0;
    struct ggml_tensor * b_0;

    // mlps_*.layers.1
    struct ggml_tensor * w_1;
    struct ggml_tensor * b_1;

    // mlps_*.layers.2
    struct ggml_tensor * w_2;
    struct ggml_tensor * b_2;
};

struct sam_decoder_mask {
    std::vector<sam_layer_dec_transformer> transformer_layers;

    // trasnformer.final_attn_token_to_image
    sam_layer_dec_transformer_attn transformer_final_attn_token_to_img;

    // transformer.norm_final
    struct ggml_tensor * transformer_norm_final_w;
    struct ggml_tensor * transformer_norm_final_b;

    // output_upscaling.0
    struct ggml_tensor * output_upscaling_0_w;
    struct ggml_tensor * output_upscaling_0_b;

    // output_upscaling.1
    struct ggml_tensor * output_upscaling_1_w;
    struct ggml_tensor * output_upscaling_1_b;

    // output_upscaling.3
    struct ggml_tensor * output_upscaling_3_w;
    struct ggml_tensor * output_upscaling_3_b;

    // output_hypernetworks_mlps
    std::vector<sam_layer_dec_output_hypernet_mlps> output_hypernet_mlps;

    // iou_prediction_head.0
    struct ggml_tensor * iou_prediction_head_0_w;
    struct ggml_tensor * iou_prediction_head_0_b;

    // iou_prediction_head.1
    struct ggml_tensor * iou_prediction_head_1_w;
    struct ggml_tensor * iou_prediction_head_1_b;

    // iou_prediction_head.2
    struct ggml_tensor * iou_prediction_head_2_w;
    struct ggml_tensor * iou_prediction_head_2_b;

    // iou_token.weight
    struct ggml_tensor * iou_token_w;

    // mask_tokens.weight
    struct ggml_tensor * mask_tokens_w;
};


struct sam_state {
    struct ggml_tensor * embd_img;

    struct ggml_tensor * low_res_masks;
    struct ggml_tensor * iou_predictions;

    //struct ggml_tensor * tmp_save = {};

    struct ggml_context * ctx;

    // buffer for `ggml_graph_plan.work_data`
    std::vector<uint8_t> work_buffer;
    // buffers to evaluate the model
    std::vector<uint8_t> buf_compute_img_enc;

    std::vector<uint8_t> buf_compute_fast;

    ggml_gallocr_t       allocr = {};
};

// void save_tensor(sam_state& state, struct ggml_tensor * t, struct ggml_cgraph * gf) {
//     if (!state.tmp_save) {
//         state.tmp_save = ggml_new_tensor(state.ctx, t->type, t->n_dims, t->ne);
//     }
//     struct ggml_tensor * tmp0 = ggml_cpy(state.ctx, t, state.tmp_save);
//     ggml_build_forward_expand(gf, tmp0);
// }

struct sam_model {
    sam_hparams hparams;

    sam_encoder_image  enc_img;
    sam_encoder_prompt enc_prompt;
    sam_decoder_mask   dec;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct sam_point {
    float x;
    float y;
};

// RGB uint8 image
struct sam_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> data;
};

// RGB float32 image
// Memory layout: RGBRGBRGB...
struct sam_image_f32 {
    int nx;
    int ny;

    std::vector<float> data;
};

struct sam_params {
    int32_t seed      = -1; // RNG seed
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());

    std::string model     = "models/sam-vit-b/ggml-model-f16.bin"; // model path
    std::string fname_inp = "img.jpg";
    std::string fname_out = "img.out";
    float   mask_threshold            = 0.f;
    float   iou_threshold             = 0.88f;
    float   stability_score_threshold = 0.95f;
    float   stability_score_offset    = 1.0f;
    float   eps                       = 1e-6f;
    float   eps_decoder_transformer   = 1e-5f;
    sam_point pt = { 414.375f, 162.796875f, };
};

void print_t_f32(const char* title, struct ggml_tensor * t, int n = 10) {
    printf("%s\n", title);
    float * data = (float *)t->data;
    printf("dims: % " PRId64 " % " PRId64 " % " PRId64 " % " PRId64 " f32\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    printf("First & Last %d elements:\n", n);
    for (int i = 0; i < std::min((int) (t->ne[0]*t->ne[1]), n); i++) {
        printf("%.5f ", data[i]);
        if (i != 0 && i % t->ne[0] == 0) {
            printf("\n");
        }
    }
    printf("\n");
    for (int i = 0; i < std::min((int) (t->ne[0]*t->ne[1]), n); i++) {
        printf("%.5f ", data[ggml_nelements(t) - n + i]);
        if ((ggml_nelements(t) - n + i) % t->ne[0] == 0) {
            printf("\n");
        }
    }
    printf("\n");
    double sum = 0.0;
    for (int i = 0; i < ggml_nelements(t); i++) {
        sum += data[i];
    }
    printf("sum:  %f\n\n", sum);
}

static void ggml_disconnect_node_from_graph(ggml_tensor * t) {
    t->op = GGML_OP_NONE;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        t->src[i] = NULL;
    }
}

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

static void ggml_sam_sin(struct ggml_tensor * dst , const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    GGML_ASSERT(userdata == NULL);
    GGML_ASSERT(ggml_are_same_shape(dst, src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src));

    const float * src_data = ggml_get_data_f32(src);
    float * dst_data = ggml_get_data_f32(dst);

    const int ne = (int)ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = std::min(ie0 + dr, ne);

    for (int i = ie0; i < ie1; ++i) {
        dst_data[i] = sinf(src_data[i]);
    }
}

static void ggml_sam_cos(struct ggml_tensor * dst , const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    GGML_ASSERT(userdata == NULL);
    GGML_ASSERT(ggml_are_same_shape(dst, src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src));

    const float * src_data = ggml_get_data_f32(src);
    float * dst_data = ggml_get_data_f32(dst);

    const int ne = (int)ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = std::min(ie0 + dr, ne);

    for (int i = ie0; i < ie1; ++i) {
        dst_data[i] = cosf(src_data[i]);
    }
}

bool sam_image_load_from_file(const std::string & fname, sam_image_u8 & img) {
    int nx, ny, nc;
    auto data = stbi_load(fname.c_str(), &nx, &ny, &nc, 3);
    if (!data) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, fname.c_str());
        return false;
    }

    img.nx = nx;
    img.ny = ny;
    img.data.resize(nx * ny * 3);
    memcpy(img.data.data(), data, nx * ny * 3);

    stbi_image_free(data);

    return true;
}

// ref: https://github.com/facebookresearch/segment-anything/blob/efeab7296ab579d4a261e554eca80faf6b33924a/segment_anything/modeling/sam.py#L164
// resize largest dimension to 1024
// normalize: x = (x - mean) / std
//     mean = [123.675, 116.28, 103.53]
//     std  = [58.395, 57.12, 57.375]
//     TODO: why are these hardcoded !?
// pad to 1024x1024
// TODO: for some reason, this is not numerically identical to pytorch's interpolation
bool sam_image_preprocess(const sam_image_u8 & img, sam_image_f32 & res) {
    const int nx = img.nx;
    const int ny = img.ny;

    const int nx2 = 1024;
    const int ny2 = 1024;

    res.nx = nx2;
    res.ny = ny2;
    res.data.resize(3*nx2*ny2);

    const float scale = std::max(nx, ny) / 1024.0f;

    fprintf(stderr, "%s: scale = %f\n", __func__, scale);

    const int nx3 = int(nx/scale + 0.5f);
    const int ny3 = int(ny/scale + 0.5f);

    const float m3[3] = { 123.675f, 116.280f, 103.530f };
    const float s3[3] = {  58.395f,  57.120f,  57.375f };

    for (int y = 0; y < ny3; y++) {
        for (int x = 0; x < nx3; x++) {
            for (int c = 0; c < 3; c++) {
                // linear interpolation
                const float sx = (x + 0.5f)*scale - 0.5f;
                const float sy = (y + 0.5f)*scale - 0.5f;

                const int x0 = std::max(0, (int) std::floor(sx));
                const int y0 = std::max(0, (int) std::floor(sy));

                const int x1 = std::min(x0 + 1, nx - 1);
                const int y1 = std::min(y0 + 1, ny - 1);

                const float dx = sx - x0;
                const float dy = sy - y0;

                const int j00 = 3*(y0*nx + x0) + c;
                const int j01 = 3*(y0*nx + x1) + c;
                const int j10 = 3*(y1*nx + x0) + c;
                const int j11 = 3*(y1*nx + x1) + c;

                const float v00 = img.data[j00];
                const float v01 = img.data[j01];
                const float v10 = img.data[j10];
                const float v11 = img.data[j11];

                const float v0 = v00*(1.0f - dx) + v01*dx;
                const float v1 = v10*(1.0f - dx) + v11*dx;

                const float v = v0*(1.0f - dy) + v1*dy;

                const uint8_t v2 = std::min(std::max(std::round(v), 0.0f), 255.0f);

                const int i = 3*(y*nx3 + x) + c;

                res.data[i] = (float(v2) - m3[c]) / s3[c];
            }
        }
    }

    return true;
}

// load the model's weights from a file
bool sam_model_load(const sam_params & params, sam_model & model) {
    fprintf(stderr, "%s: loading model from '%s' - please wait ...\n", __func__, params.model.c_str());

    auto fin = std::ifstream(params.model, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, params.model.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, params.model.c_str());
            return false;
        }
    }

    // load hparams
    {
        // Override defaults with user choices
        model.hparams.mask_threshold            = params.mask_threshold;
        model.hparams.iou_threshold             = params.iou_threshold;
        model.hparams.stability_score_threshold = params.stability_score_threshold;
        model.hparams.stability_score_offset    = params.stability_score_offset;
        model.hparams.eps                       = params.eps;
        model.hparams.eps_decoder_transformer   = params.eps_decoder_transformer;

        auto & hparams = model.hparams;

        fin.read((char *) &hparams.n_enc_state,     sizeof(hparams.n_enc_state));
        fin.read((char *) &hparams.n_enc_layer,     sizeof(hparams.n_enc_layer));
        fin.read((char *) &hparams.n_enc_head,      sizeof(hparams.n_enc_head));
        fin.read((char *) &hparams.n_enc_out_chans, sizeof(hparams.n_enc_out_chans));
        fin.read((char *) &hparams.n_pt_embd,       sizeof(hparams.n_pt_embd));
        fin.read((char *) &hparams.ftype,           sizeof(hparams.ftype));

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: n_enc_state      = %d\n", __func__, hparams.n_enc_state);
        printf("%s: n_enc_layer      = %d\n", __func__, hparams.n_enc_layer);
        printf("%s: n_enc_head       = %d\n", __func__, hparams.n_enc_head);
        printf("%s: n_enc_out_chans  = %d\n", __func__, hparams.n_enc_out_chans);
        printf("%s: n_pt_embd        = %d\n", __func__, hparams.n_pt_embd);
        printf("%s: ftype            = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr            = %d\n", __func__, qntvr);

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;

    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype) (model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
                __func__, params.model.c_str(), model.hparams.ftype);
        return false;
    }

    auto & ctx = model.ctx;

    const size_t ctx_size = [&]() {
        size_t ctx_size = 0;

        const auto & hparams = model.hparams;

        const int32_t n_enc_state     = hparams.n_enc_state;
        const int32_t n_enc_layer     = hparams.n_enc_layer;
        const int32_t n_enc_head_dim  = hparams.n_enc_head_dim();
        const int32_t n_enc_out_chans = hparams.n_enc_out_chans;
        const int32_t n_pt_embd       = hparams.n_pt_embd;

        const int32_t n_enc_layer_local  = hparams.global_attn_indices().size();
        const int32_t n_enc_layer_global = n_enc_layer - n_enc_layer_local;

        const int32_t n_img_embd    = hparams.n_img_embd();
        const int32_t n_window_size = hparams.n_window_size();
        const int32_t n_patch_size  = hparams.n_patch_size();

        // image encoder
        {
            ctx_size += n_enc_state*n_img_embd*n_img_embd*ggml_type_size(GGML_TYPE_F32);

            ctx_size += n_enc_state*3*n_patch_size*n_patch_size*ggml_type_size(GGML_TYPE_F16);
            ctx_size += n_enc_state*ggml_type_size(GGML_TYPE_F32);

            ctx_size +=     n_enc_state*n_enc_out_chans*1*1*ggml_type_size(GGML_TYPE_F16);
            ctx_size += n_enc_out_chans*n_enc_out_chans*3*3*ggml_type_size(GGML_TYPE_F16);

            ctx_size += n_enc_out_chans*ggml_type_size(GGML_TYPE_F32);
            ctx_size += n_enc_out_chans*ggml_type_size(GGML_TYPE_F32);

            ctx_size += n_enc_out_chans*ggml_type_size(GGML_TYPE_F32);
            ctx_size += n_enc_out_chans*ggml_type_size(GGML_TYPE_F32);
        }

        // image encoder layers
        {
            ctx_size += n_enc_layer*n_enc_state*ggml_type_size(GGML_TYPE_F32);
            ctx_size += n_enc_layer*n_enc_state*ggml_type_size(GGML_TYPE_F32);

            ctx_size += n_enc_layer_global*n_enc_head_dim*(2*n_img_embd - 1)*ggml_type_size(GGML_TYPE_F16);
            ctx_size += n_enc_layer_global*n_enc_head_dim*(2*n_img_embd - 1)*ggml_type_size(GGML_TYPE_F16);

            ctx_size += n_enc_layer_local*n_enc_head_dim*(2*n_window_size - 1)*ggml_type_size(GGML_TYPE_F16);
            ctx_size += n_enc_layer_local*n_enc_head_dim*(2*n_window_size - 1)*ggml_type_size(GGML_TYPE_F16);

            ctx_size += n_enc_layer*3*n_enc_state*n_enc_state*ggml_type_size(GGML_TYPE_F16);
            ctx_size += n_enc_layer*3*n_enc_state*            ggml_type_size(GGML_TYPE_F32);

            ctx_size += n_enc_layer*n_enc_state*n_enc_state*ggml_type_size(GGML_TYPE_F16);
            ctx_size += n_enc_layer*n_enc_state*            ggml_type_size(GGML_TYPE_F32);

            ctx_size += n_enc_layer*n_enc_state*ggml_type_size(GGML_TYPE_F32);
            ctx_size += n_enc_layer*n_enc_state*ggml_type_size(GGML_TYPE_F32);

            ctx_size += n_enc_layer*4*n_enc_state*n_enc_state*ggml_type_size(GGML_TYPE_F16);
            ctx_size += n_enc_layer*4*n_enc_state*            ggml_type_size(GGML_TYPE_F32);

            ctx_size += n_enc_layer*4*n_enc_state*n_enc_state*ggml_type_size(GGML_TYPE_F16);
            ctx_size += n_enc_layer*4*n_enc_state*            ggml_type_size(GGML_TYPE_F32);
        }

        ctx_size += (8 + 14*n_enc_layer)*ggml_tensor_overhead();

        // prompt encoder
        {
            ctx_size += n_enc_out_chans*ggml_type_size(GGML_TYPE_F16); // 2*(n_enc_out_chans/2)

            ctx_size += n_enc_out_chans*ggml_type_size(GGML_TYPE_F32);
            ctx_size += n_pt_embd*n_enc_out_chans*ggml_type_size(GGML_TYPE_F32);
        }

        ctx_size += (2 + n_pt_embd)*ggml_tensor_overhead();

        // mask decoder
        {
            //transformer
            {
                const int tfm_layers_count = 2;
                const int qkv_count = 3;
                const int norm_count = 4;
                const int n_hypernet_mpls_count = 4;

                // self_attn
                ctx_size += tfm_layers_count*qkv_count*n_enc_state*n_enc_state*ggml_type_size(GGML_TYPE_F16);
                ctx_size += tfm_layers_count*qkv_count*n_enc_state*            ggml_type_size(GGML_TYPE_F32);
                ctx_size += tfm_layers_count*n_enc_state*                      ggml_type_size(GGML_TYPE_F32);

                // all norms
                ctx_size += tfm_layers_count*norm_count*n_enc_state*ggml_type_size(GGML_TYPE_F32);
                ctx_size += tfm_layers_count*norm_count*n_enc_state*ggml_type_size(GGML_TYPE_F32);

                // cross_attn_token_to_img
                ctx_size += tfm_layers_count*qkv_count*n_enc_state*(n_enc_state/2)*ggml_type_size(GGML_TYPE_F16);
                ctx_size += tfm_layers_count*qkv_count*(n_enc_state/2)*            ggml_type_size(GGML_TYPE_F32);
                ctx_size += tfm_layers_count*n_enc_state*                          ggml_type_size(GGML_TYPE_F32);

                // mlp
                ctx_size += tfm_layers_count*8*n_enc_out_chans*n_enc_out_chans*ggml_type_size(GGML_TYPE_F16);
                ctx_size += tfm_layers_count*8*n_enc_out_chans*                ggml_type_size(GGML_TYPE_F32);
                ctx_size += tfm_layers_count*n_enc_out_chans*8*n_enc_out_chans*ggml_type_size(GGML_TYPE_F16);
                ctx_size += tfm_layers_count*n_enc_out_chans*                  ggml_type_size(GGML_TYPE_F32);

                // cross_attn_img_to_token
                ctx_size += tfm_layers_count*qkv_count*n_enc_state*(n_enc_state/2)*ggml_type_size(GGML_TYPE_F16);
                ctx_size += tfm_layers_count*qkv_count*(n_enc_state/2)*            ggml_type_size(GGML_TYPE_F32);
                ctx_size += tfm_layers_count*n_enc_state*                          ggml_type_size(GGML_TYPE_F32);

                // transformer_final_attn_token_to_img
                ctx_size += qkv_count*n_enc_state*(n_enc_state/2)*ggml_type_size(GGML_TYPE_F16);
                ctx_size += qkv_count*(n_enc_state/2)*            ggml_type_size(GGML_TYPE_F32);
                ctx_size += n_enc_state*                          ggml_type_size(GGML_TYPE_F32);

                // transformer_norm_final
                ctx_size += norm_count*n_enc_state*ggml_type_size(GGML_TYPE_F32);
                ctx_size += norm_count*n_enc_state*ggml_type_size(GGML_TYPE_F32);

                // output_upscaling
                ctx_size += n_enc_out_chans*n_img_embd*2*2*ggml_type_size(GGML_TYPE_F16);
                ctx_size += 3*n_img_embd*                  ggml_type_size(GGML_TYPE_F32);
                ctx_size += n_enc_out_chans*n_img_embd*(n_img_embd/2)*2*2*ggml_type_size(GGML_TYPE_F16);
                ctx_size += (n_img_embd/2)*                               ggml_type_size(GGML_TYPE_F32);

                // output_hypernetworks_mlps
                ctx_size += n_hypernet_mpls_count*2*n_enc_out_chans*n_enc_out_chans*ggml_type_size(GGML_TYPE_F16);
                ctx_size += n_hypernet_mpls_count*2*n_enc_out_chans*                ggml_type_size(GGML_TYPE_F32);
                ctx_size += n_hypernet_mpls_count*n_enc_out_chans*(n_img_embd/2)*ggml_type_size(GGML_TYPE_F16);
                ctx_size += n_hypernet_mpls_count*(n_img_embd/2)*                ggml_type_size(GGML_TYPE_F32);

                // iou_prediction_head
                ctx_size += 2*n_enc_out_chans*n_enc_out_chans*ggml_type_size(GGML_TYPE_F16);
                ctx_size += 2*n_enc_out_chans*                ggml_type_size(GGML_TYPE_F32);
                ctx_size += n_pt_embd*n_enc_out_chans*ggml_type_size(GGML_TYPE_F16);
                ctx_size += n_pt_embd*                ggml_type_size(GGML_TYPE_F32);

                // iou_token_w
                ctx_size += n_enc_out_chans*ggml_type_size(GGML_TYPE_F32);

                // mask_tokens_w
                ctx_size += n_pt_embd*n_enc_out_chans*ggml_type_size(GGML_TYPE_F32);
            }
        }
        fprintf(stderr, "%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));

        return ctx_size;
    }();

    // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };

        ctx = ggml_init(params);
        if (!ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int32_t n_enc_state      = hparams.n_enc_state;
        const int32_t n_enc_layer      = hparams.n_enc_layer;
        const int32_t n_enc_head_dim   = hparams.n_enc_head_dim();
        const int32_t n_enc_out_chans  = hparams.n_enc_out_chans;
        const int32_t n_pt_embd        = hparams.n_pt_embd;

        const int32_t n_img_embd    = hparams.n_img_embd();
        const int32_t n_window_size = hparams.n_window_size();
        const int32_t n_patch_size  = hparams.n_patch_size();

        model.enc_img.layers.resize(n_enc_layer);

        // image encoder
        {
            auto & enc = model.enc_img;

            enc.pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_enc_state, n_img_embd, n_img_embd, 1);

            enc.proj_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_patch_size, n_patch_size,           3, n_enc_state);
            enc.proj_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,            1,            1, n_enc_state);

            enc.neck_conv_0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, n_enc_state,     n_enc_out_chans);
            enc.neck_conv_1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, n_enc_out_chans, n_enc_out_chans);

            enc.neck_norm_0_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
            enc.neck_norm_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

            enc.neck_norm_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
            enc.neck_norm_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

            model.tensors["image_encoder.pos_embed"] = enc.pe;

            model.tensors["image_encoder.patch_embed.proj.weight"] = enc.proj_w;
            model.tensors["image_encoder.patch_embed.proj.bias"]   = enc.proj_b;

            model.tensors["image_encoder.neck.0.weight"] = enc.neck_conv_0;
            model.tensors["image_encoder.neck.2.weight"] = enc.neck_conv_1;

            model.tensors["image_encoder.neck.1.weight"] = enc.neck_norm_0_w;
            model.tensors["image_encoder.neck.1.bias"]   = enc.neck_norm_0_b;

            model.tensors["image_encoder.neck.3.weight"] = enc.neck_norm_1_w;
            model.tensors["image_encoder.neck.3.bias"]   = enc.neck_norm_1_b;

            for (int i = 0; i < n_enc_layer; ++i) {
                auto & layer = enc.layers[i];

                layer.norm1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_state);
                layer.norm1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_state);

                if (hparams.is_global_attn(i)) {
                    layer.rel_pos_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_head_dim, 2*n_img_embd - 1);
                    layer.rel_pos_h = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_head_dim, 2*n_img_embd - 1);
                } else {
                    layer.rel_pos_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_head_dim, 2*n_window_size - 1);
                    layer.rel_pos_h = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_head_dim, 2*n_window_size - 1);
                }

                layer.qkv_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,   n_enc_state, 3*n_enc_state);
                layer.qkv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3*n_enc_state);

                layer.proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,  n_enc_state,   n_enc_state);
                layer.proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,  n_enc_state);

                layer.norm2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_state);
                layer.norm2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_state);

                layer.mlp_lin1_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,   n_enc_state, 4*n_enc_state);
                layer.mlp_lin1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*n_enc_state);

                layer.mlp_lin2_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4*n_enc_state,   n_enc_state);
                layer.mlp_lin2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_enc_state);

                model.tensors["image_encoder.blocks." + std::to_string(i) + ".norm1.weight"] = layer.norm1_w;
                model.tensors["image_encoder.blocks." + std::to_string(i) + ".norm1.bias"]   = layer.norm1_b;

                model.tensors["image_encoder.blocks." + std::to_string(i) + ".attn.rel_pos_w"] = layer.rel_pos_w;
                model.tensors["image_encoder.blocks." + std::to_string(i) + ".attn.rel_pos_h"] = layer.rel_pos_h;

                model.tensors["image_encoder.blocks." + std::to_string(i) + ".attn.qkv.weight"] = layer.qkv_w;
                model.tensors["image_encoder.blocks." + std::to_string(i) + ".attn.qkv.bias"]   = layer.qkv_b;

                model.tensors["image_encoder.blocks." + std::to_string(i) + ".attn.proj.weight"] = layer.proj_w;
                model.tensors["image_encoder.blocks." + std::to_string(i) + ".attn.proj.bias"]   = layer.proj_b;

                model.tensors["image_encoder.blocks." + std::to_string(i) + ".norm2.weight"] = layer.norm2_w;
                model.tensors["image_encoder.blocks." + std::to_string(i) + ".norm2.bias"]   = layer.norm2_b;

                model.tensors["image_encoder.blocks." + std::to_string(i) + ".mlp.lin1.weight"] = layer.mlp_lin1_w;
                model.tensors["image_encoder.blocks." + std::to_string(i) + ".mlp.lin1.bias"]   = layer.mlp_lin1_b;

                model.tensors["image_encoder.blocks." + std::to_string(i) + ".mlp.lin2.weight"] = layer.mlp_lin2_w;
                model.tensors["image_encoder.blocks." + std::to_string(i) + ".mlp.lin2.bias"]   = layer.mlp_lin2_b;
            }
        }

        // prompt encoder
        {
            auto & enc = model.enc_prompt;

            enc.pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_enc_out_chans/2, 2);

            enc.not_a_pt_embd_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
            enc.no_mask_embd_w  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

            model.tensors["prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"] = enc.pe;
            model.tensors["prompt_encoder.not_a_point_embed.weight"] = enc.not_a_pt_embd_w;
            model.tensors["prompt_encoder.no_mask_embed.weight"]     = enc.no_mask_embd_w;

            enc.pt_embd.resize(n_pt_embd);
            for (int i = 0; i < n_pt_embd; i++) {
                enc.pt_embd[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                model.tensors["prompt_encoder.point_embeddings." + std::to_string(i) + ".weight"] = enc.pt_embd[i];
            }
        }

        // mask decoder
        {
            auto & dec = model.dec;
            auto & tfm_layers = dec.transformer_layers;

            const int tfm_layers_count = 2;
            tfm_layers.resize(tfm_layers_count);
            for (int i = 0; i < tfm_layers_count; ++i) {
                auto& l = tfm_layers[i];
                l.self_attn.q_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
                l.self_attn.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                l.self_attn.k_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
                l.self_attn.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                l.self_attn.v_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
                l.self_attn.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                l.self_attn.out_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
                l.self_attn.out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.norm1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                l.norm1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.cross_attn_token_to_img.q_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans/2);
                l.cross_attn_token_to_img.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans/2);
                l.cross_attn_token_to_img.k_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans/2);
                l.cross_attn_token_to_img.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans/2);
                l.cross_attn_token_to_img.v_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans/2);
                l.cross_attn_token_to_img.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans/2);
                l.cross_attn_token_to_img.out_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans/2, n_enc_out_chans);
                l.cross_attn_token_to_img.out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.norm2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                l.norm2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.mlp_lin1_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, 8*n_enc_out_chans);
                l.mlp_lin1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8*n_enc_out_chans);
                l.mlp_lin2_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8*n_enc_out_chans, n_enc_out_chans);
                l.mlp_lin2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.norm3_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                l.norm3_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.norm4_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                l.norm4_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.cross_attn_img_to_token.q_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans/2);
                l.cross_attn_img_to_token.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans/2);
                l.cross_attn_img_to_token.k_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans/2);
                l.cross_attn_img_to_token.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans/2);
                l.cross_attn_img_to_token.v_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans/2);
                l.cross_attn_img_to_token.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans/2);
                l.cross_attn_img_to_token.out_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans/2, n_enc_out_chans);
                l.cross_attn_img_to_token.out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                const auto prefix = "mask_decoder.transformer.layers." + std::to_string(i) + ".";
                model.tensors[prefix + "self_attn.q_proj.weight"] = l.self_attn.q_w;
                model.tensors[prefix + "self_attn.q_proj.bias"]   = l.self_attn.q_b;
                model.tensors[prefix + "self_attn.k_proj.weight"] = l.self_attn.k_w;
                model.tensors[prefix + "self_attn.k_proj.bias"]   = l.self_attn.k_b;
                model.tensors[prefix + "self_attn.v_proj.weight"] = l.self_attn.v_w;
                model.tensors[prefix + "self_attn.v_proj.bias"]   = l.self_attn.v_b;
                model.tensors[prefix + "self_attn.out_proj.weight"] = l.self_attn.out_w;
                model.tensors[prefix + "self_attn.out_proj.bias"]   = l.self_attn.out_b;

                model.tensors[prefix + "norm1.weight"] = l.norm1_w;
                model.tensors[prefix + "norm1.bias"]   = l.norm1_b;

                model.tensors[prefix + "cross_attn_token_to_image.q_proj.weight"] = l.cross_attn_token_to_img.q_w;
                model.tensors[prefix + "cross_attn_token_to_image.q_proj.bias"]   = l.cross_attn_token_to_img.q_b;
                model.tensors[prefix + "cross_attn_token_to_image.k_proj.weight"] = l.cross_attn_token_to_img.k_w;
                model.tensors[prefix + "cross_attn_token_to_image.k_proj.bias"]   = l.cross_attn_token_to_img.k_b;
                model.tensors[prefix + "cross_attn_token_to_image.v_proj.weight"] = l.cross_attn_token_to_img.v_w;
                model.tensors[prefix + "cross_attn_token_to_image.v_proj.bias"]   = l.cross_attn_token_to_img.v_b;
                model.tensors[prefix + "cross_attn_token_to_image.out_proj.weight"] = l.cross_attn_token_to_img.out_w;
                model.tensors[prefix + "cross_attn_token_to_image.out_proj.bias"]   = l.cross_attn_token_to_img.out_b;

                model.tensors[prefix + "norm2.weight"] = l.norm2_w;
                model.tensors[prefix + "norm2.bias"]   = l.norm2_b;

                model.tensors[prefix + "mlp.lin1.weight"] = l.mlp_lin1_w;
                model.tensors[prefix + "mlp.lin1.bias"]   = l.mlp_lin1_b;
                model.tensors[prefix + "mlp.lin2.weight"] = l.mlp_lin2_w;
                model.tensors[prefix + "mlp.lin2.bias"]   = l.mlp_lin2_b;

                model.tensors[prefix + "norm3.weight"] = l.norm3_w;
                model.tensors[prefix + "norm3.bias"]   = l.norm3_b;
                model.tensors[prefix + "norm4.weight"] = l.norm4_w;
                model.tensors[prefix + "norm4.bias"]   = l.norm4_b;

                model.tensors[prefix + "cross_attn_image_to_token.q_proj.weight"] = l.cross_attn_img_to_token.q_w;
                model.tensors[prefix + "cross_attn_image_to_token.q_proj.bias"]   = l.cross_attn_img_to_token.q_b;
                model.tensors[prefix + "cross_attn_image_to_token.k_proj.weight"] = l.cross_attn_img_to_token.k_w;
                model.tensors[prefix + "cross_attn_image_to_token.k_proj.bias"]   = l.cross_attn_img_to_token.k_b;
                model.tensors[prefix + "cross_attn_image_to_token.v_proj.weight"] = l.cross_attn_img_to_token.v_w;
                model.tensors[prefix + "cross_attn_image_to_token.v_proj.bias"]   = l.cross_attn_img_to_token.v_b;
                model.tensors[prefix + "cross_attn_image_to_token.out_proj.weight"] = l.cross_attn_img_to_token.out_w;
                model.tensors[prefix + "cross_attn_image_to_token.out_proj.bias"]   = l.cross_attn_img_to_token.out_b;
            }

            dec.transformer_final_attn_token_to_img.q_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans/2);
            dec.transformer_final_attn_token_to_img.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans/2);
            dec.transformer_final_attn_token_to_img.k_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans/2);
            dec.transformer_final_attn_token_to_img.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans/2);
            dec.transformer_final_attn_token_to_img.v_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans/2);
            dec.transformer_final_attn_token_to_img.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans/2);
            dec.transformer_final_attn_token_to_img.out_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans/2, n_enc_out_chans);
            dec.transformer_final_attn_token_to_img.out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

            model.tensors["mask_decoder.transformer.final_attn_token_to_image.q_proj.weight"] = dec.transformer_final_attn_token_to_img.q_w;
            model.tensors["mask_decoder.transformer.final_attn_token_to_image.q_proj.bias"]   = dec.transformer_final_attn_token_to_img.q_b;
            model.tensors["mask_decoder.transformer.final_attn_token_to_image.k_proj.weight"] = dec.transformer_final_attn_token_to_img.k_w;
            model.tensors["mask_decoder.transformer.final_attn_token_to_image.k_proj.bias"]   = dec.transformer_final_attn_token_to_img.k_b;
            model.tensors["mask_decoder.transformer.final_attn_token_to_image.v_proj.weight"] = dec.transformer_final_attn_token_to_img.v_w;
            model.tensors["mask_decoder.transformer.final_attn_token_to_image.v_proj.bias"]   = dec.transformer_final_attn_token_to_img.v_b;
            model.tensors["mask_decoder.transformer.final_attn_token_to_image.out_proj.weight"] = dec.transformer_final_attn_token_to_img.out_w;
            model.tensors["mask_decoder.transformer.final_attn_token_to_image.out_proj.bias"]   = dec.transformer_final_attn_token_to_img.out_b;

            dec.transformer_norm_final_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
            dec.transformer_norm_final_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

            model.tensors["mask_decoder.transformer.norm_final_attn.weight"] = dec.transformer_norm_final_w;
            model.tensors["mask_decoder.transformer.norm_final_attn.bias"]   = dec.transformer_norm_final_b;

            dec.output_upscaling_0_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 2, 2, n_img_embd, n_enc_out_chans);
            dec.output_upscaling_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_img_embd);
            dec.output_upscaling_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_img_embd);
            dec.output_upscaling_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_img_embd);
            dec.output_upscaling_3_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16,  2, 2, n_img_embd/2, n_img_embd);
            dec.output_upscaling_3_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_img_embd/2);

            model.tensors["mask_decoder.output_upscaling.0.weight"] = dec.output_upscaling_0_w;
            model.tensors["mask_decoder.output_upscaling.0.bias"]   = dec.output_upscaling_0_b;
            model.tensors["mask_decoder.output_upscaling.1.weight"] = dec.output_upscaling_1_w;
            model.tensors["mask_decoder.output_upscaling.1.bias"]   = dec.output_upscaling_1_b;
            model.tensors["mask_decoder.output_upscaling.3.weight"] = dec.output_upscaling_3_w;
            model.tensors["mask_decoder.output_upscaling.3.bias"]   = dec.output_upscaling_3_b;

            const int n_hypernet_mpls_count = 4;
            dec.output_hypernet_mlps.resize(n_hypernet_mpls_count);
            for (int i = 0; i < n_hypernet_mpls_count; ++i) {
                auto& mlp = dec.output_hypernet_mlps[i];

                mlp.w_0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
                mlp.b_0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                mlp.w_1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
                mlp.b_1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                mlp.w_2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_img_embd/2);
                mlp.b_2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_img_embd/2);

                const auto prefix = "mask_decoder.output_hypernetworks_mlps." + std::to_string(i) + ".";
                model.tensors[prefix + "layers.0.weight"] = mlp.w_0;
                model.tensors[prefix + "layers.0.bias"]   = mlp.b_0;
                model.tensors[prefix + "layers.1.weight"] = mlp.w_1;
                model.tensors[prefix + "layers.1.bias"]   = mlp.b_1;
                model.tensors[prefix + "layers.2.weight"] = mlp.w_2;
                model.tensors[prefix + "layers.2.bias"]   = mlp.b_2;
            }

            dec.iou_prediction_head_0_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
            dec.iou_prediction_head_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
            dec.iou_prediction_head_1_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
            dec.iou_prediction_head_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
            dec.iou_prediction_head_2_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_pt_embd);
            dec.iou_prediction_head_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_pt_embd);

            dec.iou_token_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_enc_out_chans, 1);
            dec.mask_tokens_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_enc_out_chans, n_pt_embd);

            model.tensors["mask_decoder.iou_prediction_head.layers.0.weight"] = dec.iou_prediction_head_0_w;
            model.tensors["mask_decoder.iou_prediction_head.layers.0.bias"]   = dec.iou_prediction_head_0_b;
            model.tensors["mask_decoder.iou_prediction_head.layers.1.weight"] = dec.iou_prediction_head_1_w;
            model.tensors["mask_decoder.iou_prediction_head.layers.1.bias"]   = dec.iou_prediction_head_1_b;
            model.tensors["mask_decoder.iou_prediction_head.layers.2.weight"] = dec.iou_prediction_head_2_w;
            model.tensors["mask_decoder.iou_prediction_head.layers.2.bias"]   = dec.iou_prediction_head_2_b;

            model.tensors["mask_decoder.iou_token.weight"] = dec.iou_token_w;
            model.tensors["mask_decoder.mask_tokens.weight"] = dec.mask_tokens_w;
        }
    }

    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;

        fprintf(stderr, "%s: ", __func__);

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

            if (fin.eof()) {
                break;
            }

            int64_t nelements = 1;
            int64_t ne[4] = { 1, 1, 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                int32_t ne_cur;
                fin.read(reinterpret_cast<char *>(&ne_cur), sizeof(ne_cur));
                ne[i] = ne_cur;
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            //printf("ne0 = %jd, ne1 = %jd, ne2 = %jd, ne3 = %jd\n", ne[0], ne[1], ne[2], ne[3]);

            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %d, expected %d\n",
                        __func__, name.data(), (int) nelements, (int) ggml_nelements(tensor));
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2] || tensor->ne[3] != ne[3]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d, %d, %d], expected [%d, %d, %d, %d]\n",
                        __func__, name.data(),
                        (int) ne[0], (int) ne[1], (int) ne[2], (int) ne[3],
                        (int) tensor->ne[0], (int) tensor->ne[1], (int) tensor->ne[2], (int) tensor->ne[3]);
                return false;
            }

            size_t bpe = 0;

            switch (ftype) {
                case 0: bpe = ggml_type_size(GGML_TYPE_F32);  break;
                case 1: bpe = ggml_type_size(GGML_TYPE_F16);  break;
                case 2: bpe = ggml_type_size(GGML_TYPE_Q4_0); assert(ne[0] % 64 == 0); break;
                case 3: bpe = ggml_type_size(GGML_TYPE_Q4_1); assert(ne[0] % 64 == 0); break;
                default:
                        {
                            fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                            return false;
                        }
            };

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), (size_t) nelements*bpe);
                return false;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 8 == 0) {
                fprintf(stderr, ".");
                fflush(stdout);
            }
        }

        if (n_tensors != int(model.tensors.size())) {
            fprintf(stderr, "%s: model file has %d tensors, but %d tensors were expected\n", __func__, n_tensors, (int) model.tensors.size());
            return false;
        }

        fprintf(stderr, " done\n");

        fprintf(stderr, "%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size/1024.0/1024.0, n_tensors);
    }

    fin.close();

    return true;
}

struct ggml_tensor * sam_fill_dense_pe(
            const sam_model   & model,
          struct ggml_context * ctx0,
          struct ggml_cgraph  * gf,
                  sam_state   & state) {
    const auto & hparams = model.hparams;
    const auto & enc     = model.enc_prompt;


    const int32_t n_img_embd = hparams.n_img_embd();
    struct ggml_tensor * xy_embed_stacked = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 2, n_img_embd, n_img_embd);
    ggml_set_name(xy_embed_stacked, "xy_embed_stacked");
    ggml_set_input(xy_embed_stacked);

    struct ggml_tensor * cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, enc.pe)), xy_embed_stacked);

    cur = ggml_scale(ctx0, cur, float(2.0*M_PI));

    // concat
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L192
    {
        struct ggml_tensor * t_sin = ggml_map_custom1(ctx0, cur, ggml_sam_sin, GGML_N_TASKS_MAX, NULL);
        struct ggml_tensor * t_cos = ggml_map_custom1(ctx0, cur, ggml_sam_cos, GGML_N_TASKS_MAX, NULL);

        cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, t_sin->ne[0] + t_cos->ne[0], cur->ne[1], cur->ne[2]);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, t_sin, ggml_view_3d(ctx0, cur, t_sin->ne[0], t_sin->ne[1], t_sin->ne[2], cur->nb[1], cur->nb[2], 0)));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, t_cos, ggml_view_3d(ctx0, cur, t_sin->ne[0], t_sin->ne[1], t_sin->ne[2], cur->nb[1], cur->nb[2], t_sin->nb[1])));
    }

    struct ggml_tensor * pe_img_dense = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));
    ggml_build_forward_expand(gf, pe_img_dense);

    return pe_img_dense;
}

struct ggml_tensor* sam_layer_norm_2d(
                    struct ggml_context * ctx0,
                    struct ggml_tensor  * layer,
                    int                   n_channels,
                    struct ggml_tensor  * w,
                    struct ggml_tensor  * b,
                    float                 eps) {
    // LayerNorm2d
    // normalize along channel dimmension
    // TODO: better implementation
    layer = ggml_permute(ctx0,
                ggml_norm(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, layer, 1, 2, 0, 3)), eps),
                2, 0, 1, 3);

    layer = ggml_add(ctx0,
              ggml_mul(ctx0,
                  ggml_repeat(ctx0, ggml_reshape_3d(ctx0, w, 1, 1, n_channels), layer),
                  layer),
              ggml_repeat(ctx0, ggml_reshape_3d(ctx0, b, 1, 1, n_channels), layer));

    return layer;
}

struct ggml_cgraph  * sam_encode_image(
            const sam_model & model,
                  sam_state & state,
        const sam_image_f32 & img) {

    const auto & hparams = model.hparams;
    const auto & enc     = model.enc_img;

    const int32_t n_enc_state     = hparams.n_enc_state;
    const int32_t n_enc_layer     = hparams.n_enc_layer;
    const int32_t n_enc_head      = hparams.n_enc_head;
    const int32_t n_enc_head_dim  = hparams.n_enc_head_dim();
    const int32_t n_enc_out_chans = hparams.n_enc_out_chans;
    const int32_t n_img_size    = hparams.n_img_size();
    const int32_t n_window_size = hparams.n_window_size();

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ state.buf_compute_img_enc.size(),
        /*.mem_buffer =*/ state.buf_compute_img_enc.data(),
        /*.no_alloc   =*/ true, // skip allocating as we use ggml_alloc to allocate exact memory requirements
    };

    struct ggml_context * ctx0   = ggml_init(ggml_params);
    struct ggml_cgraph  * gf     = ggml_new_graph(ctx0);

    struct ggml_tensor * inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_img_size, n_img_size, 3, 1);
    ggml_set_name(inp, "inp");
    ggml_set_input(inp);

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L392
    struct ggml_tensor * cur = ggml_conv_2d_sk_p0(ctx0, enc.proj_w, inp);
    cur = ggml_add_inplace(ctx0,
            cur,
            ggml_repeat(ctx0, enc.proj_b, cur));

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L394
    // keep in F32
    cur = ggml_cont(ctx0,
            ggml_permute(ctx0, cur, 1, 2, 0, 3));

    // convert to F16
    //cur = ggml_cpy(ctx0,
    //        ggml_permute(ctx0, cur, 1, 2, 0, 3),
    //        ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, n_enc_state, n_img_embd, n_img_embd));

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L108-L109
    cur = ggml_add_inplace(ctx0, cur, enc.pe);

    struct ggml_tensor * inpL = cur;

    for (int il = 0; il < n_enc_layer; ++il) {
        const auto & layer = enc.layers[il];

        // norm
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L168
        {
            cur = ggml_norm(ctx0, inpL, hparams.eps);

            // cur = ln_0_w*cur + ln_0_b
            cur = ggml_mul(ctx0, cur, layer.norm1_w);
            cur = ggml_add_inplace(ctx0, cur, layer.norm1_b);
        }

        const int64_t w0 = cur->ne[1];
        const int64_t h0 = cur->ne[2];

        if (hparams.is_global_attn(il) == false) {
            // local attention layer - apply window partition
            // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L169-L172
            cur = ggml_win_part(ctx0, cur, n_window_size);
        }

        const int64_t W = cur->ne[1];
        const int64_t H = cur->ne[2];

        // self-attention
        {
            cur = ggml_mul_mat(ctx0, layer.qkv_w, cur);
            cur = ggml_add_inplace(ctx0, cur, layer.qkv_b);

            // split qkv into separate tensors
            // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L225-L229
            const int B = cur->ne[3];

            cur = ggml_reshape_4d(ctx0, cur, n_enc_state, 3, W*H, B);
            cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 3, 1, 2));

            struct ggml_tensor * Q;
            struct ggml_tensor * K;
            struct ggml_tensor * V;

            Q = ggml_view_3d   (ctx0, cur, n_enc_state, W*H, B, cur->nb[1], cur->nb[2], 0*cur->nb[3]);
            Q = ggml_reshape_4d(ctx0, Q,   n_enc_head_dim, n_enc_head, W*H, B);
            Q = ggml_cont      (ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(ctx0, Q,   n_enc_head_dim, W*H, B*n_enc_head);

            K = ggml_view_3d   (ctx0, cur, n_enc_state, W*H, B, cur->nb[1], cur->nb[2], 1*cur->nb[3]);
            K = ggml_reshape_4d(ctx0, K,   n_enc_head_dim, n_enc_head, W*H, B);
            K = ggml_cont      (ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            K = ggml_reshape_3d(ctx0, K,   n_enc_head_dim, W*H, B*n_enc_head);

            V = ggml_view_3d   (ctx0, cur, n_enc_state, W*H, B, cur->nb[1], cur->nb[2], 2*cur->nb[3]);
            V = ggml_reshape_4d(ctx0, V,   n_enc_head_dim, n_enc_head, W*H, B);
            V = ggml_cont      (ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3)); // transposed
            V = ggml_reshape_3d(ctx0, V,   W*H, n_enc_head_dim, B*n_enc_head);

            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            struct ggml_tensor * KQ_scaled =
                ggml_scale_inplace(ctx0,
                        KQ,
                        1.0f/sqrtf(n_enc_head_dim));

            struct ggml_tensor * rw = ggml_get_rel_pos(ctx0, layer.rel_pos_w, W, W);
            struct ggml_tensor * rh = ggml_get_rel_pos(ctx0, layer.rel_pos_h, H, H);

            struct ggml_tensor * q_r = ggml_reshape_4d(ctx0, Q, n_enc_head_dim, W, H, B*n_enc_head);

            struct ggml_tensor * rel_w = ggml_cont(ctx0, ggml_permute(ctx0,
                        ggml_mul_mat(ctx0,
                            rw,
                            ggml_cont(ctx0, ggml_permute(ctx0, q_r, 0, 2, 1, 3))),
                        0, 2, 1, 3));
            struct ggml_tensor * rel_h = ggml_mul_mat(ctx0, rh, q_r);

            struct ggml_tensor * attn = ggml_add_rel_pos_inplace(ctx0, KQ_scaled, rel_w, rel_h);

            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, attn);

            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

            cur =
                ggml_reshape_4d(ctx0,
                        ggml_cont(ctx0,
                            ggml_permute(ctx0,
                                ggml_reshape_4d(ctx0, KQV, n_enc_head_dim, W*H, n_enc_head, B),
                                0, 2, 1, 3)),
                        n_enc_state, W, H, B);

            cur = ggml_mul_mat(ctx0, layer.proj_w, cur);
            cur = ggml_add_inplace(ctx0, cur, layer.proj_b);
        }

        if (hparams.is_global_attn(il) == false) {
            // local attention layer - reverse window partition
            cur = ggml_win_unpart(ctx0, cur, w0, h0, n_window_size);
        }

        cur = ggml_add_inplace(ctx0, cur, inpL);

        struct ggml_tensor * inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF, hparams.eps);

                // cur = mlp_ln_w*cur + mlp_ln_b
                cur = ggml_mul(ctx0, cur, layer.norm2_w);
                cur = ggml_add_inplace(ctx0, cur, layer.norm2_b);
            }

            // fully connected
            cur = ggml_mul_mat(ctx0, layer.mlp_lin1_w, cur);
            cur = ggml_add_inplace(ctx0, cur, layer.mlp_lin1_b);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // projection
            cur = ggml_mul_mat(ctx0, layer.mlp_lin2_w, cur);
            cur = ggml_add_inplace(ctx0, cur, layer.mlp_lin2_b);
        }

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    cur = ggml_cont(ctx0, ggml_permute(ctx0, inpL, 2, 0, 1, 3));

    cur = ggml_conv_2d_sk_p0(ctx0, enc.neck_conv_0, cur);

    cur = sam_layer_norm_2d(ctx0, cur, n_enc_out_chans, enc.neck_norm_0_w, enc.neck_norm_0_b, hparams.eps);

    cur = ggml_conv_2d_s1_ph(ctx0, enc.neck_conv_1, cur);

    cur = sam_layer_norm_2d(ctx0, cur, n_enc_out_chans, enc.neck_norm_1_w, enc.neck_norm_1_b, hparams.eps);

    cur = ggml_cpy(ctx0, cur, state.embd_img);

    ggml_build_forward_expand(gf, cur);
    ggml_disconnect_node_from_graph(state.embd_img);

    //ggml_graph_print(&gf);

    ggml_free(ctx0);

    ggml_gallocr_alloc_graph(state.allocr, gf);

    {
        struct ggml_tensor * inp = ggml_graph_get_tensor(gf, "inp");
        float * data = (float *) ggml_get_data(inp);

        const int nx = img.nx;
        const int ny = img.ny;
        const int n  = nx*ny;

        GGML_ASSERT(nx == n_img_size && ny == n_img_size);

        for (int k = 0; k < 3; k++) {
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    data[k*n + y*nx + x] = img.data[3*(y*nx + x) + k];
                }
            }
        }
    }

    return gf;
}


struct prompt_encoder_result {
    struct ggml_tensor * embd_prompt_sparse = {};
    struct ggml_tensor * embd_prompt_dense = {};
};

// encode a prompt
//
// - points
// - boxes
// - masks
//
// TODO: currently just encode a single point for simplicity
//
prompt_encoder_result sam_encode_prompt(
        const sam_model     & model,
        struct ggml_context * ctx0,
        struct ggml_cgraph  * gf,
                  sam_state & state) {

    const auto & hparams = model.hparams;
    const auto & enc = model.enc_prompt;

    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2, 2);
    ggml_set_name(inp, "prompt_input");
    ggml_set_input(inp);


    struct ggml_tensor * cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, enc.pe)), inp);

    cur = ggml_scale(ctx0, cur, float(2.0*M_PI));

    // concat
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L192
    {
        struct ggml_tensor * t_sin = ggml_map_custom1(ctx0, cur, ggml_sam_sin, GGML_N_TASKS_MAX, NULL);
        struct ggml_tensor * t_cos = ggml_map_custom1(ctx0, cur, ggml_sam_cos, GGML_N_TASKS_MAX, NULL);

        cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, t_sin->ne[0] + t_cos->ne[0], cur->ne[1]);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, t_sin, ggml_view_2d(ctx0, cur, t_sin->ne[0], t_sin->ne[1], cur->nb[1], 0)));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, t_cos, ggml_view_2d(ctx0, cur, t_sin->ne[0], t_sin->ne[1], cur->nb[1], t_sin->nb[1])));

        // overwrite label == -1 with not_a_point_embed.weight
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L86
        // TODO: extend for multiple points
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, enc.not_a_pt_embd_w, ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], cur->nb[1])));
    }

    // add point_embeddings[1] to label == 1
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L90
    struct ggml_tensor * v = ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], 0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_add_inplace(ctx0, v, enc.pt_embd[1]), v));

    struct ggml_tensor * embd_prompt_sparse = cur;
    ggml_build_forward_expand(gf, embd_prompt_sparse);

    struct ggml_tensor * embd_prompt_dense = ggml_repeat(ctx0,
            ggml_cont(ctx0,
                ggml_view_3d(ctx0, enc.no_mask_embd_w,
                    1, 1, enc.no_mask_embd_w->ne[0], enc.no_mask_embd_w->nb[0], enc.no_mask_embd_w->nb[0], 0)),
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hparams.n_img_embd(), hparams.n_img_embd(), hparams.n_enc_out_chans));

    ggml_build_forward_expand(gf, embd_prompt_dense);

    //printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    prompt_encoder_result res;
    res.embd_prompt_sparse = embd_prompt_sparse;
    res.embd_prompt_dense  = embd_prompt_dense;
    return res;
}

struct ggml_tensor* sam_decode_mask_transformer_attn(
    const sam_layer_dec_transformer_attn & attn,
                      struct ggml_tensor * queries,
                      struct ggml_tensor * keys,
                      struct ggml_tensor * values,
                     struct ggml_context * ctx0,
                         const sam_model & model) {
    const auto & hparams = model.hparams;
    const int n_head = hparams.n_dec_heads;

    struct ggml_tensor * Qcur = {};
    struct ggml_tensor * Kcur = {};
    struct ggml_tensor * Vcur = {};

    Qcur = ggml_mul_mat(ctx0, attn.q_w, queries);
    Qcur = ggml_add_inplace(ctx0, Qcur, attn.q_b);

    Kcur = ggml_mul_mat(ctx0, attn.k_w, keys);
    Kcur = ggml_add_inplace(ctx0, Kcur, attn.k_b);

    Vcur = ggml_mul_mat(ctx0, attn.v_w, values);
    Vcur = ggml_add_inplace(ctx0, Vcur, attn.v_b);

    struct ggml_tensor * Q = {};
    struct ggml_tensor * K = {};
    struct ggml_tensor * V = {};

    Q = ggml_reshape_4d(ctx0, Qcur, Qcur->ne[0]/n_head, n_head, Qcur->ne[1], Qcur->ne[2]);
    Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));

    K = ggml_reshape_4d(ctx0, Kcur, Kcur->ne[0]/n_head, n_head, Kcur->ne[1], Kcur->ne[2]);
    K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));

    V = ggml_reshape_4d(ctx0, Vcur, Vcur->ne[0]/n_head, n_head, Vcur->ne[1], Vcur->ne[2]);
    V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

    // Q * K
    struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

    struct ggml_tensor * KQ_scaled = ggml_scale_inplace(ctx0, KQ, 1.0f/sqrt(float(Q->ne[0])));

    struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_scaled);

    struct ggml_tensor * KQV = ggml_mul_mat(ctx0, KQ_soft_max, ggml_cont(ctx0, ggml_transpose(ctx0, V)));

    struct ggml_tensor * KQV_merged = ggml_cont(ctx0, ggml_transpose(ctx0, KQV));
    KQV_merged = ggml_cont(ctx0, ggml_permute(ctx0, KQV_merged, 0, 2, 1, 3));
    KQV_merged = ggml_reshape_3d(ctx0, KQV_merged, KQV_merged->ne[0]*KQV_merged->ne[1], KQV_merged->ne[2], KQV_merged->ne[3]);
    KQV_merged = ggml_mul_mat(ctx0, attn.out_w, KQV_merged);
    KQV_merged = ggml_add_inplace(ctx0, KQV_merged, attn.out_b);

    return KQV_merged;
}

struct ggml_tensor * sam_decode_mask_mlp_relu_3(
     struct ggml_tensor * in,
     struct ggml_tensor * w_0,
     struct ggml_tensor * b_0,
     struct ggml_tensor * w_1,
     struct ggml_tensor * b_1,
     struct ggml_tensor * w_2,
     struct ggml_tensor * b_2,
    struct ggml_context * ctx0) {

    struct ggml_tensor * cur = {};
    cur = ggml_mul_mat(ctx0, w_0, in);
    cur = ggml_add_inplace(ctx0, cur, b_0);

    cur = ggml_relu_inplace(ctx0, cur);

    cur = ggml_mul_mat(ctx0, w_1, cur);
    cur = ggml_add_inplace(ctx0, cur, b_1);

    cur = ggml_relu_inplace(ctx0, cur);

    cur = ggml_mul_mat(ctx0, w_2, cur);
    cur = ggml_add_inplace(ctx0, cur, b_2);

    return cur;
}

bool sam_decode_mask(
                    const sam_model & model,
        const prompt_encoder_result & prompt,
                 struct ggml_tensor * pe_img,
                struct ggml_context * ctx0,
                struct ggml_cgraph  * gf,
                          sam_state & state) {

    const auto & hparams = model.hparams;
    const auto & dec = model.dec;
    const int n_img_embd = hparams.n_img_embd();

    struct ggml_tensor * tokens = {};
    {
        // Concatenate output tokens
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L120
        const auto& sparse = prompt.embd_prompt_sparse;

        tokens = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, dec.iou_token_w->ne[0], dec.iou_token_w->ne[1] + dec.mask_tokens_w->ne[1] + sparse->ne[1], sparse->ne[2]);

        const size_t offsets[3] = { 0, dec.iou_token_w->ne[1]*tokens->nb[1], dec.iou_token_w->ne[1]*tokens->nb[1] + dec.mask_tokens_w->ne[1]*tokens->nb[1] };
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, dec.iou_token_w,   ggml_view_2d(ctx0, tokens, tokens->ne[0], dec.iou_token_w->ne[1],   tokens->nb[1], offsets[0])));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, dec.mask_tokens_w, ggml_view_2d(ctx0, tokens, tokens->ne[0], dec.mask_tokens_w->ne[1], tokens->nb[1], offsets[1])));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, sparse,            ggml_view_2d(ctx0, tokens, tokens->ne[0], sparse->ne[1],            tokens->nb[1], offsets[2])));
        // TODO: Sparse prompt embeddings can have more than one point
    }


    struct ggml_tensor * src = {};
    struct ggml_tensor * pos_src = {};
    int srcNE[4] = { 0, 0, 0, 0 };
    {
        // Expand per-image data in the batch direction to be per-mask
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L125
        src = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, state.embd_img->ne[0], state.embd_img->ne[1], state.embd_img->ne[2], tokens->ne[2]);

        src = ggml_add(ctx0,
            ggml_repeat(ctx0,
                state.embd_img,
                src),
            prompt.embd_prompt_dense);

        srcNE[0] = src->ne[0];
        srcNE[1] = src->ne[1];
        srcNE[2] = src->ne[2];
        srcNE[3] = src->ne[3];

        // flatten & permute
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L83
        src = ggml_cont(ctx0, ggml_permute(ctx0,
            ggml_view_3d(ctx0,
                src,
                src->ne[0]*src->ne[1],
                src->ne[2],
                src->ne[3],
                src->nb[2],
                src->nb[3],
                0),
            1, 0, 2, 3));

        pos_src = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, pe_img->ne[0], pe_img->ne[1], pe_img->ne[2], tokens->ne[2]);
        pos_src = ggml_repeat(ctx0,
            pe_img,
            pos_src);

        // flatten & permute
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L83
        pos_src = ggml_cont(ctx0, ggml_permute(ctx0,
            ggml_view_3d(ctx0,
                pos_src,
                pos_src->ne[0]*pos_src->ne[1],
                pos_src->ne[2],
                pos_src->ne[3],
                pos_src->nb[2],
                pos_src->nb[3],
                0),
            1, 0, 2, 3));
    }

    struct ggml_tensor * queries = tokens;
    struct ggml_tensor * keys = src;
    {
        // Run the transformer
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L62
        for (int i = 0; i < int(model.dec.transformer_layers.size()); ++i) {
            const auto& tfm_layer = model.dec.transformer_layers[i];

            // Self attention block
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L154
            const bool skip_first_layer_pe = i == 0;
            if (skip_first_layer_pe) {
                queries = sam_decode_mask_transformer_attn(tfm_layer.self_attn, queries, queries, queries, ctx0, model);
            }
            else {
                struct ggml_tensor * q_0 = ggml_add(ctx0, queries, tokens);

                struct ggml_tensor * self_attn = sam_decode_mask_transformer_attn(tfm_layer.self_attn, q_0, q_0, queries, ctx0, model);
                queries = ggml_add(ctx0, queries, self_attn);
            }

            queries = ggml_norm(ctx0, queries, hparams.eps_decoder_transformer);
            queries = ggml_add_inplace(ctx0,
                    ggml_mul(ctx0, queries, tfm_layer.norm1_w),
                    tfm_layer.norm1_b);

            // Cross attention block, tokens attending to image embedding
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L163
            struct ggml_tensor * q_1 = ggml_add(ctx0, queries, tokens);
            struct ggml_tensor * k_1 = ggml_add(ctx0, keys, pos_src);

            struct ggml_tensor * cross_attn_token_to_img = sam_decode_mask_transformer_attn(tfm_layer.cross_attn_token_to_img, q_1, k_1, keys, ctx0, model);

            queries = ggml_add_inplace(ctx0, queries, cross_attn_token_to_img);
            queries = ggml_norm_inplace(ctx0, queries, hparams.eps_decoder_transformer);
            queries = ggml_add_inplace(ctx0,
                    ggml_mul(ctx0, queries, tfm_layer.norm2_w),
                    tfm_layer.norm2_b);

            // MLP block
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L170
            struct ggml_tensor * mlp_out = ggml_mul_mat(ctx0,
                tfm_layer.mlp_lin1_w,
                queries);

            mlp_out = ggml_add_inplace(ctx0, mlp_out, tfm_layer.mlp_lin1_b);

            // RELU activation
            mlp_out = ggml_relu_inplace(ctx0, mlp_out);
            mlp_out = ggml_mul_mat(ctx0, tfm_layer.mlp_lin2_w, mlp_out);

            mlp_out = ggml_add_inplace(ctx0, mlp_out, tfm_layer.mlp_lin2_b);

            queries = ggml_add_inplace(ctx0, queries, mlp_out);
            queries = ggml_norm_inplace(ctx0, queries, hparams.eps_decoder_transformer);
            queries = ggml_add_inplace(ctx0,
                    ggml_mul(ctx0, queries, tfm_layer.norm3_w),
                    tfm_layer.norm3_b);

            // Cross attention block, image embedding attending to tokens
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L175
            struct ggml_tensor * q_2 = ggml_add(ctx0, queries, tokens);
            struct ggml_tensor * k_2 = ggml_add(ctx0, keys, pos_src);

            struct ggml_tensor * cross_attn_img_to_token = sam_decode_mask_transformer_attn(tfm_layer.cross_attn_img_to_token, k_2, q_2, queries, ctx0, model);
            keys = ggml_add_inplace(ctx0, keys, cross_attn_img_to_token);
            keys = ggml_norm_inplace(ctx0, keys, hparams.eps_decoder_transformer);
            keys = ggml_add_inplace(ctx0,
                    ggml_mul(ctx0, keys, tfm_layer.norm4_w),
                    tfm_layer.norm4_b);
        }

        // Apply the final attention layer from the points to the image
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L99
        struct ggml_tensor * q = ggml_add(ctx0, queries, tokens);
        struct ggml_tensor * k = ggml_add(ctx0, keys, pos_src);

        struct ggml_tensor * final_attn_token_to_img = sam_decode_mask_transformer_attn(dec.transformer_final_attn_token_to_img, q, k, keys, ctx0, model);

        queries = ggml_add_inplace(ctx0, queries, final_attn_token_to_img);
        queries = ggml_norm_inplace(ctx0, queries, hparams.eps_decoder_transformer);
        queries = ggml_add_inplace(ctx0,
                ggml_mul(ctx0, queries, dec.transformer_norm_final_w),
                dec.transformer_norm_final_b);
    }


    struct ggml_tensor * iou_pred = ggml_view_2d(ctx0, queries, queries->ne[0], queries->ne[2], queries->nb[2], 0);
    const int num_mask_tokens = 4; // num_multimask_outputs + 1
    struct ggml_tensor * mask_tokens_out = ggml_view_3d(ctx0, queries, queries->ne[0], num_mask_tokens, queries->ne[2], queries->nb[1], num_mask_tokens*queries->nb[1], queries->nb[1]);

    // Upscale mask embeddings and predict masks using the mask tokens
    // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L136
    keys = ggml_cont(ctx0, ggml_transpose(ctx0, keys));
    keys = ggml_view_4d(ctx0, keys, srcNE[0], srcNE[1], srcNE[2], srcNE[3], srcNE[0]*keys->nb[0], keys->nb[1], keys->nb[2], 0);
    // ggml_build_forward_expand(gf, keys);
    struct ggml_tensor * upscaled_embedding = {};
    {
        // ConvTranspose2d
        keys = ggml_conv_transpose_2d_p0(ctx0, dec.output_upscaling_0_w, keys, 2);
        keys = ggml_add_inplace(ctx0, keys, ggml_repeat(ctx0,
                                     ggml_reshape_3d(ctx0, dec.output_upscaling_0_b, 1, 1, dec.output_upscaling_0_b->ne[0]),
                                     keys));

        keys = sam_layer_norm_2d(ctx0, keys, n_img_embd, dec.output_upscaling_1_w, dec.output_upscaling_1_b, hparams.eps);

        // GELU activation
        keys = ggml_gelu_inplace(ctx0, keys);

        // ConvTranspose2d
        keys = ggml_conv_transpose_2d_p0(ctx0, dec.output_upscaling_3_w, keys, 2);
        keys = ggml_add_inplace(ctx0, ggml_repeat(ctx0,
                                ggml_reshape_3d(ctx0, dec.output_upscaling_3_b, 1, 1, dec.output_upscaling_3_b->ne[0]),
                                keys), keys);
        // GELU activation
        keys = ggml_gelu_inplace(ctx0, keys);
        upscaled_embedding = ggml_reshape_3d(ctx0, keys, keys->ne[0]*keys->ne[1], keys->ne[2], keys->ne[3]);
        upscaled_embedding = ggml_cont(ctx0, ggml_transpose(ctx0, upscaled_embedding)); // TODO: Shouldn't be needed
    }

    struct ggml_tensor * hyper_in = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_img_embd/2, num_mask_tokens, mask_tokens_out->ne[2]);

    for (int i = 0; i < num_mask_tokens; ++i) {
        const auto& mlp = dec.output_hypernet_mlps[i];
        struct ggml_tensor * in = ggml_view_2d(ctx0, mask_tokens_out, mask_tokens_out->ne[0], mask_tokens_out->ne[2], mask_tokens_out->nb[1], i*mask_tokens_out->nb[1]);
        struct ggml_tensor * out = sam_decode_mask_mlp_relu_3(in, mlp.w_0, mlp.b_0, mlp.w_1, mlp.b_1, mlp.w_2, mlp.b_2, ctx0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, out, ggml_view_2d(ctx0, hyper_in, hyper_in->ne[0], hyper_in->ne[2], hyper_in->nb[1], i*hyper_in->nb[1])));
    }

    struct ggml_tensor * masks = ggml_mul_mat(ctx0, hyper_in, upscaled_embedding);
    masks = ggml_cont(ctx0, ggml_transpose(ctx0, masks)); // TODO: Shouldn't be needed
    masks = ggml_reshape_4d(ctx0, masks, keys->ne[0], keys->ne[1], masks->ne[1], keys->ne[3]);

    // Generate mask quality predictions
    // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L146
    iou_pred = sam_decode_mask_mlp_relu_3(iou_pred, dec.iou_prediction_head_0_w, dec.iou_prediction_head_0_b, dec.iou_prediction_head_1_w, dec.iou_prediction_head_1_b, dec.iou_prediction_head_2_w, dec.iou_prediction_head_2_b, ctx0);

    // Select the correct mask or masks for output
    // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L101
    iou_pred = ggml_cpy(state.ctx, ggml_view_1d(ctx0, iou_pred, iou_pred->ne[0] - 1, iou_pred->nb[0]), state.iou_predictions);
    masks = ggml_view_4d(ctx0, masks, masks->ne[0], masks->ne[1], masks->ne[2] - 1, masks->ne[3],
                                      masks->nb[1], masks->nb[2], masks->nb[3], masks->nb[2] /* offset*/);
    masks = ggml_cpy(state.ctx, masks, state.low_res_masks);

    ggml_build_forward_expand(gf, masks);
    ggml_build_forward_expand(gf, iou_pred);

    ggml_disconnect_node_from_graph(state.low_res_masks);
    ggml_disconnect_node_from_graph(state.iou_predictions);

    return true;
}

bool sam_write_masks(const sam_hparams& hparams, int nx, int ny, const sam_state & state, const std::string & fname) {
    if (state.low_res_masks->ne[2] == 0) return true;
    if (state.low_res_masks->ne[2] != state.iou_predictions->ne[0]) {
        printf("Error: number of masks (%d) does not match number of iou predictions (%d)\n", (int)state.low_res_masks->ne[2], (int)state.iou_predictions->ne[0]);
        return false;
    }

    const int n_img_size = hparams.n_img_size();
    const float mask_threshold = hparams.mask_threshold;
    const float iou_threshold = hparams.iou_threshold;
    const float stability_score_threshold = hparams.stability_score_threshold;
    const float intersection_threshold = mask_threshold + hparams.stability_score_offset;
    const float union_threshold = mask_threshold - hparams.stability_score_offset;

    const int ne0 = state.low_res_masks->ne[0];
    const int ne1 = state.low_res_masks->ne[1];
    const int ne2 = state.low_res_masks->ne[2];

    // Remove padding and upscale masks to the original image size.
    // ref: https://github.com/facebookresearch/segment-anything/blob/efeab7296ab579d4a261e554eca80faf6b33924a/segment_anything/modeling/sam.py#L140

    const float preprocess_scale = std::max(nx, ny) / float(n_img_size);
    const int cropped_nx = int(nx / preprocess_scale + 0.5f);
    const int cropped_ny = int(ny / preprocess_scale + 0.5f);

    const float scale_x_1 = (float)ne0 / (float)n_img_size;
    const float scale_y_1 = (float)ne1 / (float)n_img_size;

    const float scale_x_2 = float(cropped_nx) / float(nx);
    const float scale_y_2 = float(cropped_ny) / float(ny);

    const auto iou_data = (float*)state.iou_predictions->data;

    for (int i = 0; i < ne2; ++i) {
        if (iou_threshold > 0.f && iou_data[i] < iou_threshold) {
            printf("Skipping mask %d with iou %f below threshold %f\n", i, iou_data[i], iou_threshold);
            continue; // Filtering masks with iou below the threshold
        }

        std::vector<float> mask_data(n_img_size*n_img_size);
        {
            const float* data = (float *) state.low_res_masks->data + i*ne0*ne1;

            for (int iy = 0; iy < n_img_size; ++iy) {
                for (int ix = 0; ix < n_img_size; ++ix) {
                    const float sx = std::max(scale_x_1*(ix + 0.5f) - 0.5f, 0.0f);
                    const float sy = std::max(scale_y_1*(iy + 0.5f) - 0.5f, 0.0f);

                    const int x0 = std::max(0, (int)sx);
                    const int y0 = std::max(0, (int)sy);

                    const int x1 = std::min(x0 + 1, ne0 - 1);
                    const int y1 = std::min(y0 + 1, ne1 - 1);

                    const float dx = sx - x0;
                    const float dy = sy - y0;

                    const int j00 = y0*ne0 + x0;
                    const int j01 = y0*ne0 + x1;
                    const int j10 = y1*ne0 + x0;
                    const int j11 = y1*ne0 + x1;

                    const float v00 = data[j00];
                    const float v01 = data[j01];
                    const float v10 = data[j10];
                    const float v11 = data[j11];

                    const float v0 = (1-dx)*v00 + dx*v01;
                    const float v1 = (1-dx)*v10 + dx*v11;

                    const float v = (1-dy)*v0 + dy*v1;

                    mask_data[iy*n_img_size + ix] = v;
                }
            }
        }

        int intersections = 0;
        int unions = 0;
        sam_image_u8 res;
        int min_iy = ny;
        int max_iy = 0;
        int min_ix = nx;
        int max_ix = 0;
        {
            const float* data = mask_data.data();

            res.nx = nx;
            res.ny = ny;
            res.data.resize(nx*ny);

            for (int iy = 0; iy < ny; ++iy) {
                for (int ix = 0; ix < nx; ++ix) {
                    const float sx = std::max(scale_x_2*(ix + 0.5f) - 0.5f, 0.0f);
                    const float sy = std::max(scale_y_2*(iy + 0.5f) - 0.5f, 0.0f);

                    const int x0 = std::max(0, (int)sx);
                    const int y0 = std::max(0, (int)sy);

                    const int x1 = std::min(x0 + 1, cropped_nx - 1);
                    const int y1 = std::min(y0 + 1, cropped_ny - 1);

                    const float dx = sx - x0;
                    const float dy = sy - y0;

                    const int j00 = y0*n_img_size + x0;
                    const int j01 = y0*n_img_size + x1;
                    const int j10 = y1*n_img_size + x0;
                    const int j11 = y1*n_img_size + x1;

                    const float v00 = data[j00];
                    const float v01 = data[j01];
                    const float v10 = data[j10];
                    const float v11 = data[j11];

                    const float v0 = (1-dx)*v00 + dx*v01;
                    const float v1 = (1-dx)*v10 + dx*v11;

                    const float v = (1-dy)*v0 + dy*v1;

                    if (v > intersection_threshold) {
                        intersections++;
                    }
                    if (v > union_threshold) {
                        unions++;
                    }
                    if (v > mask_threshold) {
                        min_iy = std::min(min_iy, iy);
                        max_iy = std::max(max_iy, iy);
                        min_ix = std::min(min_ix, ix);
                        max_ix = std::max(max_ix, ix);

                        res.data[iy*nx + ix] = 255;
                    }
                }
            }
        }

        const float stability_score = float(intersections) / float(unions);
        if (stability_score_threshold > 0.f && stability_score < stability_score_threshold) {
            printf("Skipping mask %d with stability score %f below threshold %f\n", i, stability_score, stability_score_threshold);
            continue; // Filtering masks with stability score below the threshold
        }

        printf("Mask %d: iou = %f, stability_score = %f, bbox (%d, %d), (%d, %d)\n",
                i, iou_data[i], stability_score, min_ix, max_ix, min_iy, max_iy);

        std::string filename = fname + std::to_string(i) + ".png";
        if (!stbi_write_png(filename.c_str(), res.nx, res.ny, 1, res.data.data(), res.nx)) {
            printf("%s: failed to write mask %s\n", __func__, filename.c_str());
            return false;
        }
    }


    return true;
}

struct ggml_cgraph  * sam_build_fast_graph(
        const sam_model     & model,
                  sam_state & state,
                        int   nx,
                        int   ny,
                  sam_point   point) {

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ state.buf_compute_fast.size(),
        /*.mem_buffer =*/ state.buf_compute_fast.data(),
        /*.no_alloc   =*/ true, // skip allocating as we use ggml_alloc to allocate exact memory requirements
    };

    struct ggml_context * ctx0   = ggml_init(ggml_params);
    struct ggml_cgraph  * gf     = ggml_new_graph(ctx0);

    prompt_encoder_result enc_res = sam_encode_prompt(model, ctx0, gf, state);
    if (!enc_res.embd_prompt_sparse || !enc_res.embd_prompt_dense) {
        fprintf(stderr, "%s: failed to encode prompt (%f, %f)\n", __func__, point.x, point.y);
        return {};
    }

    struct ggml_tensor * pe_img_dense = sam_fill_dense_pe(model, ctx0, gf, state);
    if (!pe_img_dense) {
        fprintf(stderr, "%s: failed to get dense positional encoding\n", __func__);
        return {};
    }

    if (!sam_decode_mask(model, enc_res, pe_img_dense, ctx0, gf, state)) {
         fprintf(stderr, "%s: failed to decode mask\n", __func__);
         return {};
    }

    ggml_free(ctx0);

    ggml_gallocr_alloc_graph(state.allocr, gf);

    // from sam_encode_prompt
    {
        // transform points
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py#L276
        {
            const int nmax = std::max(nx, ny);

            const float scale = model.hparams.n_img_size() / (float) nmax;

            const int nx_new = int(nx*scale + 0.5f);
            const int ny_new = int(ny*scale + 0.5f);

            point.x = point.x*(float(nx_new)/nx) + 0.5f;
            point.y = point.y*(float(ny_new)/ny) + 0.5f;
        }

        struct ggml_tensor * inp = ggml_graph_get_tensor(gf, "prompt_input");
        // set the input by converting the [0, 1] coordinates to [-1, 1]
        float * data = (float *) inp->data;

        data[0] = 2.0f*(point.x / model.hparams.n_img_size()) - 1.0f;
        data[1] = 2.0f*(point.y / model.hparams.n_img_size()) - 1.0f;

        // padding
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L81-L85
        data[2] = 2.0f*(0.0f) - 1.0f;
        data[3] = 2.0f*(0.0f) - 1.0f;
    }

    // from sam_fill_dense_pe
    {
        struct ggml_tensor * xy_embed_stacked = ggml_graph_get_tensor(gf, "xy_embed_stacked");
        const int32_t n_img_embd = model.hparams.n_img_embd();
        const float n_img_embd_inv = 1.0f / n_img_embd;
        float * data = (float *) ggml_get_data(xy_embed_stacked);
        for (int i = 0; i < n_img_embd; ++i) {
            const int row = 2*i*n_img_embd;
            const float y_val = 2 * (i + 0.5f) * n_img_embd_inv - 1;
            for (int j = 0; j < n_img_embd; ++j) {
                const float x_val = 2 * (j + 0.5f) * n_img_embd_inv - 1;
                data[row + 2*j + 0] = x_val;
                data[row + 2*j + 1] = y_val;
            }
        }
    }

    return gf;
}

void sam_print_usage(int argc, char ** argv, const sam_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  -i FNAME, --inp FNAME\n");
    fprintf(stderr, "                        input file (default: %s)\n", params.fname_inp.c_str());
    fprintf(stderr, "  -o FNAME, --out FNAME\n");
    fprintf(stderr, "                        mask file name prefix (default: %s)\n", params.fname_out.c_str());
    fprintf(stderr, "SAM hyperparameters:\n");
    fprintf(stderr, "  -mt FLOAT, --mask-threshold\n");
    fprintf(stderr, "                        mask threshold (default: %f)\n", params.mask_threshold);
    fprintf(stderr, "  -it FLOAT, --iou-threshold\n");
    fprintf(stderr, "                        iou threshold (default: %f)\n", params.iou_threshold);
    fprintf(stderr, "  -st FLOAT, --score-threshold\n");
    fprintf(stderr, "                        score threshold (default: %f)\n", params.stability_score_threshold);
    fprintf(stderr, "  -so FLOAT, --score-offset\n");
    fprintf(stderr, "                        score offset (default: %f)\n", params.stability_score_offset);
    fprintf(stderr, "  -e FLOAT, --epsilon\n");
    fprintf(stderr, "                        epsilon (default: %f)\n", params.eps);
    fprintf(stderr, "  -ed FLOAT, --epsilon-decoder-transformer\n");
    fprintf(stderr, "                        epsilon decoder transformer (default: %f)\n", params.eps_decoder_transformer);
    fprintf(stderr, "SAM prompt:\n");
    fprintf(stderr, "  -p TUPLE, --point-prompt\n");
    fprintf(stderr, "                        point to be used as prompt for SAM (default: %f,%f). Must be in a format FLOAT,FLOAT \n", params.pt.x, params.pt.y);
    fprintf(stderr, "\n");
}

bool sam_params_parse(int argc, char ** argv, sam_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-i" || arg == "--inp") {
            params.fname_inp = argv[++i];
        } else if (arg == "-o" || arg == "--out") {
            params.fname_out = argv[++i];
        } else if (arg == "-mt" || arg == "--mask-threshold") {
            params.mask_threshold = std::stof(argv[++i]);
        } else if (arg == "-it" || arg == "--iou-threshold") {
            params.iou_threshold = std::stof(argv[++i]);
        } else if (arg == "-st" || arg == "--score-threshold") {
            params.stability_score_threshold = std::stof(argv[++i]);
        } else if (arg == "-so" || arg == "--score-offset") {
            params.stability_score_offset = std::stof(argv[++i]);
        } else if (arg == "-e" || arg == "--epsilon") {
            params.eps = std::stof(argv[++i]);
        } else if (arg == "-ed" || arg == "--epsilon-decoder-transformer") {
            params.eps_decoder_transformer = std::stof(argv[++i]);
        } else if (arg == "-p" || arg == "--point-prompt") {
            // TODO multiple points per model invocation
            char* point = argv[++i];

            char* coord = strtok(point, ",");
            if (!coord){
                fprintf(stderr, "Error while parsing prompt!\n");
                exit(1);
            }
            params.pt.x = std::stof(coord);
            coord = strtok(NULL, ",");
            if (!coord){
                fprintf(stderr, "Error while parsing prompt!\n");
                exit(1);
            }
            params.pt.y = std::stof(coord);
        } else if (arg == "-h" || arg == "--help") {
            sam_print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            sam_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    const int64_t t_main_start_us = ggml_time_us();

    sam_params params;
    params.model = "models/sam-vit-b/ggml-model-f16.bin";

    sam_model model;
    sam_state state;
    int64_t t_load_us = 0;

    if (sam_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }
    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    // load the image
    sam_image_u8 img0;
    if (!sam_image_load_from_file(params.fname_inp, img0)) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.fname_inp.c_str());
        return 1;
    }
    fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, params.fname_inp.c_str(), img0.nx, img0.ny);

    // preprocess to f32
    sam_image_f32 img1;
    if (!sam_image_preprocess(img0, img1)) {
        fprintf(stderr, "%s: failed to preprocess image\n", __func__);
        return 1;
    }
    fprintf(stderr, "%s: preprocessed image (%d x %d)\n", __func__, img1.nx, img1.ny);


    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!sam_model_load(params, model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    {
        static size_t buf_size = 256u*1024*1024;

        struct ggml_init_params ggml_params = {
            /*.mem_size   =*/ buf_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };

        state.ctx = ggml_init(ggml_params);

        state.embd_img = ggml_new_tensor_3d(state.ctx, GGML_TYPE_F32,
                model.hparams.n_img_embd(), model.hparams.n_img_embd(), model.hparams.n_enc_out_chans);

        state.low_res_masks = ggml_new_tensor_3d(state.ctx, GGML_TYPE_F32,
                model.hparams.n_enc_out_chans, model.hparams.n_enc_out_chans, 3);

        state.iou_predictions = ggml_new_tensor_1d(state.ctx, GGML_TYPE_F32, 3);
    }


    {
        state.buf_compute_img_enc.resize(ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead());
        state.allocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

        struct ggml_cgraph  * gf = sam_encode_image(model, state, img1);
        if (!gf) {
            fprintf(stderr, "%s: failed to encode image\n", __func__);
            return 1;
        }

        ggml_graph_compute_helper(state.work_buffer, gf, params.n_threads);

        print_t_f32("embd_img", state.embd_img);

        ggml_gallocr_free(state.allocr);
        state.allocr = NULL;
        state.work_buffer.clear();
    }
    {
        state.buf_compute_fast.resize(ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead());
        state.allocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

        // TODO: more varied prompts
        fprintf(stderr, "prompt: (%f, %f)\n", params.pt.x, params.pt.y);

        struct ggml_cgraph  * gf = sam_build_fast_graph(model, state, img0.nx, img0.ny, params.pt);
        if (!gf) {
            fprintf(stderr, "%s: failed to build fast graph\n", __func__);
            return 1;
        }

        ggml_graph_compute_helper(state.work_buffer, gf, params.n_threads);

        //print_t_f32("iou_predictions", state.iou_predictions);
        //print_t_f32("low_res_masks", state.low_res_masks);
        ggml_gallocr_free(state.allocr);
        state.allocr = NULL;
    }

    if (!sam_write_masks(model.hparams, img0.nx, img0.ny, state, params.fname_out)) {
        fprintf(stderr, "%s: failed to write masks\n", __func__);
        return 1;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        fprintf(stderr, "\n\n");
        fprintf(stderr, "%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        fprintf(stderr, "%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}
