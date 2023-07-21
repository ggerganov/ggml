#include "ggml.h"

#include "common.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// default hparams (ViT-B SAM)
struct sam_hparams {
    int32_t n_enc_state       = 768;
    int32_t n_enc_layer       = 12;
    int32_t n_enc_head        = 12;
    int32_t n_enc_out_chans   = 256;
    int32_t n_pt_embd         = 4;
    int32_t ftype             = 1;

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
    struct ggml_tensor * mlp_ln1_w;
    struct ggml_tensor * mlp_ln1_b;

    // mlp.lin2
    struct ggml_tensor * mlp_ln2_w;
    struct ggml_tensor * mlp_ln2_b;

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
    struct ggml_tensor * embd_prompt_sparse;
    struct ggml_tensor * embd_prompt_dense;
    struct ggml_tensor * pe_img_dense;

    struct ggml_context * ctx;

    std::vector<uint8_t> buf;
};

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

void ggml_sam_sin(const int n, float * dst, const float * src) {
    for (int i = 0; i < n; ++i) {
        dst[i] = sinf(src[i]);
    }
}

void ggml_sam_cos(const int n, float * dst, const float * src) {
    for (int i = 0; i < n; ++i) {
        dst[i] = cosf(src[i]);
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
bool sam_model_load(const std::string & fname, sam_model & model) {
    fprintf(stderr, "%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // load hparams
    {
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
                __func__, fname.c_str(), model.hparams.ftype);
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
            ctx_size += n_enc_state*n_img_embd*n_img_embd*ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += n_enc_state*3*n_patch_size*n_patch_size*ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += n_enc_state*ggml_type_sizef(GGML_TYPE_F32);

            ctx_size +=     n_enc_state*n_enc_out_chans*1*1*ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += n_enc_out_chans*n_enc_out_chans*3*3*ggml_type_sizef(GGML_TYPE_F16);

            ctx_size += n_enc_out_chans*ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += n_enc_out_chans*ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += n_enc_out_chans*ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += n_enc_out_chans*ggml_type_sizef(GGML_TYPE_F32);
        }

        // image encoder layers
        {
            ctx_size += n_enc_layer*n_enc_state*ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += n_enc_layer*n_enc_state*ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += n_enc_layer_global*n_enc_head_dim*(2*n_img_embd - 1)*ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += n_enc_layer_global*n_enc_head_dim*(2*n_img_embd - 1)*ggml_type_sizef(GGML_TYPE_F16);

            ctx_size += n_enc_layer_local*n_enc_head_dim*(2*n_window_size - 1)*ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += n_enc_layer_local*n_enc_head_dim*(2*n_window_size - 1)*ggml_type_sizef(GGML_TYPE_F16);

            ctx_size += n_enc_layer*3*n_enc_state*n_enc_state*ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += n_enc_layer*3*n_enc_state*            ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += n_enc_layer*n_enc_state*n_enc_state*ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += n_enc_layer*n_enc_state*            ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += n_enc_layer*n_enc_state*ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += n_enc_layer*n_enc_state*ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += n_enc_layer*4*n_enc_state*n_enc_state*ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += n_enc_layer*4*n_enc_state*            ggml_type_sizef(GGML_TYPE_F32);

            ctx_size += n_enc_layer*4*n_enc_state*n_enc_state*ggml_type_sizef(GGML_TYPE_F16);
            ctx_size += n_enc_layer*4*n_enc_state*            ggml_type_sizef(GGML_TYPE_F32);
        }

        ctx_size += (8 + 14*n_enc_layer)*ggml_tensor_overhead();

        // prompt encoder
        {
            ctx_size += n_enc_out_chans*ggml_type_sizef(GGML_TYPE_F16); // 2*(n_enc_out_chans/2)

            ctx_size += n_enc_out_chans*ggml_type_sizef(GGML_TYPE_F32);
            ctx_size += n_pt_embd*n_enc_out_chans*ggml_type_sizef(GGML_TYPE_F32);
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
                ctx_size += tfm_layers_count*qkv_count*n_enc_state*n_enc_state*ggml_type_sizef(GGML_TYPE_F16);
                ctx_size += tfm_layers_count*qkv_count*n_enc_state*            ggml_type_sizef(GGML_TYPE_F32);
                ctx_size += tfm_layers_count*n_enc_state*                      ggml_type_sizef(GGML_TYPE_F32);

                // all norms
                ctx_size += tfm_layers_count*norm_count*n_enc_state*ggml_type_sizef(GGML_TYPE_F32);
                ctx_size += tfm_layers_count*norm_count*n_enc_state*ggml_type_sizef(GGML_TYPE_F32);

                // cross_attn_token_to_img
                ctx_size += tfm_layers_count*qkv_count*n_enc_state*(n_enc_state/2)*ggml_type_sizef(GGML_TYPE_F16);
                ctx_size += tfm_layers_count*qkv_count*(n_enc_state/2)*            ggml_type_sizef(GGML_TYPE_F32);
                ctx_size += tfm_layers_count*n_enc_state*                          ggml_type_sizef(GGML_TYPE_F32);

                // mlp
                ctx_size += tfm_layers_count*8*n_enc_out_chans*n_enc_out_chans*ggml_type_sizef(GGML_TYPE_F16);
                ctx_size += tfm_layers_count*8*n_enc_out_chans*                ggml_type_sizef(GGML_TYPE_F32);
                ctx_size += tfm_layers_count*n_enc_out_chans*8*n_enc_out_chans*ggml_type_sizef(GGML_TYPE_F16);
                ctx_size += tfm_layers_count*n_enc_out_chans*                  ggml_type_sizef(GGML_TYPE_F32);

                // cross_attn_img_to_token
                ctx_size += tfm_layers_count*qkv_count*n_enc_state*(n_enc_state/2)*ggml_type_sizef(GGML_TYPE_F16);
                ctx_size += tfm_layers_count*qkv_count*(n_enc_state/2)*            ggml_type_sizef(GGML_TYPE_F32);
                ctx_size += tfm_layers_count*n_enc_state*                          ggml_type_sizef(GGML_TYPE_F32);

                // transformer_final_attn_token_to_img
                ctx_size += qkv_count*n_enc_state*(n_enc_state/2)*ggml_type_sizef(GGML_TYPE_F16);
                ctx_size += qkv_count*(n_enc_state/2)*            ggml_type_sizef(GGML_TYPE_F32);
                ctx_size += n_enc_state*                          ggml_type_sizef(GGML_TYPE_F32);

                // transformer_norm_final
                ctx_size += norm_count*n_enc_state*ggml_type_sizef(GGML_TYPE_F32);
                ctx_size += norm_count*n_enc_state*ggml_type_sizef(GGML_TYPE_F32);

                // output_upscaling
                ctx_size += n_enc_out_chans*n_img_embd*2*2*ggml_type_sizef(GGML_TYPE_F16);
                ctx_size += 3*n_img_embd*                  ggml_type_sizef(GGML_TYPE_F32);
                ctx_size += n_enc_out_chans*n_img_embd*(n_img_embd/2)*2*2*ggml_type_sizef(GGML_TYPE_F16);
                ctx_size += (n_img_embd/2)*                               ggml_type_sizef(GGML_TYPE_F32);

                // output_hypernetworks_mlps
                ctx_size += 3*n_enc_out_chans*n_enc_out_chans*ggml_type_sizef(GGML_TYPE_F16);
                ctx_size += 3*n_enc_out_chans*                ggml_type_sizef(GGML_TYPE_F32);

                // iou_prediction_head
                ctx_size += 3*n_enc_out_chans*n_enc_out_chans*ggml_type_sizef(GGML_TYPE_F16);
                ctx_size += 3*n_enc_out_chans*                ggml_type_sizef(GGML_TYPE_F32);

                // iou_token_w
                ctx_size += n_enc_out_chans*ggml_type_sizef(GGML_TYPE_F16);

                // mask_tokens_w
                ctx_size += n_pt_embd*n_enc_out_chans*ggml_type_sizef(GGML_TYPE_F16);
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

                l.cross_attn_token_to_img.q_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans/2, n_enc_out_chans);
                l.cross_attn_token_to_img.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans/2);
                l.cross_attn_token_to_img.k_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans/2, n_enc_out_chans);
                l.cross_attn_token_to_img.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans/2);
                l.cross_attn_token_to_img.v_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans/2, n_enc_out_chans);
                l.cross_attn_token_to_img.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans/2);
                l.cross_attn_token_to_img.out_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans/2);
                l.cross_attn_token_to_img.out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.norm2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                l.norm2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.mlp_ln1_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8*n_enc_out_chans, n_enc_out_chans);
                l.mlp_ln1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8*n_enc_out_chans);
                l.mlp_ln2_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, 8*n_enc_out_chans);
                l.mlp_ln2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.norm3_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                l.norm3_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.norm4_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                l.norm4_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.cross_attn_img_to_token.q_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans/2, n_enc_out_chans);
                l.cross_attn_img_to_token.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans/2);
                l.cross_attn_img_to_token.k_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans/2, n_enc_out_chans);
                l.cross_attn_img_to_token.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans/2);
                l.cross_attn_img_to_token.v_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans/2, n_enc_out_chans);
                l.cross_attn_img_to_token.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans/2);
                l.cross_attn_img_to_token.out_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans/2);
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

                model.tensors[prefix + "mlp_lin1.weight"] = l.mlp_ln1_w;
                model.tensors[prefix + "mlp_lin1.bias"]   = l.mlp_ln1_b;
                model.tensors[prefix + "mlp_lin2.weight"] = l.mlp_ln2_w;
                model.tensors[prefix + "mlp_lin2.bias"]   = l.mlp_ln2_b;

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

            dec.transformer_final_attn_token_to_img.q_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans/2, n_enc_out_chans);
            dec.transformer_final_attn_token_to_img.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans/2);
            dec.transformer_final_attn_token_to_img.k_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans/2, n_enc_out_chans);
            dec.transformer_final_attn_token_to_img.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans/2);
            dec.transformer_final_attn_token_to_img.v_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans/2, n_enc_out_chans);
            dec.transformer_final_attn_token_to_img.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans/2);
            dec.transformer_final_attn_token_to_img.out_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans/2);
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

            model.tensors["mask_decoder.transformer.norm_final.weight"] = dec.transformer_norm_final_w;
            model.tensors["mask_decoder.transformer.norm_final.bias"]   = dec.transformer_norm_final_b;

            dec.output_upscaling_0_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_img_embd, 2, 2);
            dec.output_upscaling_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_img_embd);
            dec.output_upscaling_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_img_embd);
            dec.output_upscaling_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_img_embd);
            dec.output_upscaling_3_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_img_embd, n_img_embd/2, 2, 2);
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
                mlp.w_2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
                mlp.b_2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

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
            dec.iou_prediction_head_2_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
            dec.iou_prediction_head_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

            dec.iou_token_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 1, n_enc_out_chans);
            dec.mask_tokens_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_pt_embd, n_enc_out_chans);

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

    // key + value memory
    {
        // const auto & hparams = model.hparams;

        // TODO
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

        fprintf(stderr, " done\n");

        fprintf(stderr, "%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size/1024.0/1024.0, n_tensors);
    }

    fin.close();

    return true;
}

bool sam_fill_dense_pe(
            const sam_model & model,
                  sam_state & state,
                        int   n_threads) {
    const auto & hparams = model.hparams;
    const auto & enc     = model.enc_prompt;

    const int32_t n_img_embd = hparams.n_img_embd();
    const float n_img_embd_inv = 1.0f / n_img_embd;

    static size_t buf_size = 256u*1024*1024;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};

    struct ggml_tensor * xy_embed_stacked = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 2, n_img_embd, n_img_embd);

    {
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

    struct ggml_tensor * cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, enc.pe)), xy_embed_stacked);

    cur = ggml_scale(ctx0, cur, ggml_new_f32(ctx0, 2.0f*M_PI));

    // concat
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L192
    {
        struct ggml_tensor * t_sin = ggml_map_unary_f32(ctx0, cur, ggml_sam_sin);
        struct ggml_tensor * t_cos = ggml_map_unary_f32(ctx0, cur, ggml_sam_cos);

        cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, t_sin->ne[0] + t_cos->ne[0], cur->ne[1], cur->ne[2]);

        ggml_build_forward_expand(&gf, ggml_cpy(ctx0, t_sin, ggml_view_3d(ctx0, cur, t_sin->ne[0], t_sin->ne[1], t_sin->ne[2], cur->nb[1], cur->nb[2], 0)));
        ggml_build_forward_expand(&gf, ggml_cpy(ctx0, t_cos, ggml_view_3d(ctx0, cur, t_sin->ne[0], t_sin->ne[1], t_sin->ne[2], cur->nb[1], cur->nb[2], t_sin->nb[1])));
    }

    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));

    // TODO: avoid copy
    cur = ggml_cpy(ctx0, cur, state.pe_img_dense);

    // run the computation
    ggml_set_name(cur, "check");
    ggml_build_forward_expand(&gf, cur);
    ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

    // auto * t = ggml_get_tensor(ctx0, "check");
    // auto print_t_f32 = [&](struct ggml_tensor * t) {
    //     float * data = (float *)t->data;
    //     printf("dims: %jd %jd %jd %jd f32\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    //     for (int i = 0; i < 256; i++) {
    //         printf("%f ", data[256*64*63 + 63*256 + i]);
    //     }
    //     printf("\n");
    //     double sum = 0.0;
    //     for (int i = 0; i < ggml_nelements(t); i++) {
    //         sum += data[i];
    //     }
    //     printf("sum:  %f\n", sum);
    // };
    // print_t_f32(t);

    return true;
}

bool sam_encode_image(
            const sam_model & model,
                  sam_state & state,
        const sam_image_f32 & img,
                        int   n_threads) {

    const auto & hparams = model.hparams;
    const auto & enc     = model.enc_img;

    const int32_t n_enc_state     = hparams.n_enc_state;
    const int32_t n_enc_layer     = hparams.n_enc_layer;
    const int32_t n_enc_head      = hparams.n_enc_head;
    const int32_t n_enc_head_dim  = hparams.n_enc_head_dim();
    const int32_t n_enc_out_chans = hparams.n_enc_out_chans;

    const int32_t n_img_size    = hparams.n_img_size();
    const int32_t n_window_size = hparams.n_window_size();

    static size_t buf_size = 256u*1024*1024;
    static void * buf = malloc(buf_size);

    // use 2 scratch buffers
    // TODO: very hacky solution - reimplement in a more elegant way
    static size_t scr0_size = 2048u*1024*1024;
    static void * scr0 = malloc(scr0_size);

    static size_t scr1_size = 512u*1024*1024;
    static void * scr1 = malloc(scr1_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};

    struct ggml_tensor * inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_img_size, n_img_size, 3, 1);
    {
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

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L392
    struct ggml_tensor * cur = ggml_conv_2d_sk_p0(ctx0, enc.proj_w, inp);
    cur = ggml_add(ctx0,
            ggml_repeat(ctx0,
                enc.proj_b,
                cur),
            cur);

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L394
    // keep in F32
    cur = ggml_cont(ctx0,
            ggml_permute(ctx0, cur, 1, 2, 0, 3));

    // convert to F16
    //cur = ggml_cpy(ctx0,
    //        ggml_permute(ctx0, cur, 1, 2, 0, 3),
    //        ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, n_enc_state, n_img_embd, n_img_embd));

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L108-L109
    cur = ggml_add(ctx0, enc.pe, cur);

    struct ggml_tensor * inpL = cur;

    for (int il = 0; il < n_enc_layer; ++il) {
        const auto & layer = enc.layers[il];

        ggml_set_scratch(ctx0, { 0, scr0_size, scr0, });

        // norm
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L168
        {
            cur = ggml_norm(ctx0, inpL);

            // cur = ln_0_w*cur + ln_0_b
            cur = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, layer.norm1_w, cur),
                        cur),
                    ggml_repeat(ctx0, layer.norm1_b, cur));
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

        ggml_set_scratch(ctx0, { 0, scr1_size, scr1, });

        // self-attention
        {
            cur = ggml_mul_mat(ctx0, layer.qkv_w, cur);
            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0,
                        layer.qkv_b,
                        cur),
                    cur);

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

            ggml_set_scratch(ctx0, { 0, scr0_size, scr0, });

            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            struct ggml_tensor * KQ_scaled =
                ggml_scale_inplace(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrtf(n_enc_head_dim))
                        );

            struct ggml_tensor * rw = ggml_get_rel_pos(ctx0, layer.rel_pos_w, W, W);
            struct ggml_tensor * rh = ggml_get_rel_pos(ctx0, layer.rel_pos_h, H, H);

            struct ggml_tensor * q_r = ggml_reshape_4d(ctx0, Q, n_enc_head_dim, W, H, B*n_enc_head);

            struct ggml_tensor * rel_w = ggml_cont(ctx0, ggml_permute(ctx0,
                        ggml_mul_mat(ctx0,
                            rw,
                            ggml_cont(ctx0, ggml_permute(ctx0, q_r, 0, 2, 1, 3))),
                        0, 2, 1, 3));
            struct ggml_tensor * rel_h = ggml_mul_mat(ctx0, rh, q_r);

            struct ggml_tensor * attn = ggml_add_rel_pos(ctx0, KQ_scaled, rel_w, rel_h);

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
            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0,
                        layer.proj_b,
                        cur),
                    cur);
        }

        if (hparams.is_global_attn(il) == false) {
            // local attention layer - reverse window partition
            cur = ggml_win_unpart(ctx0, cur, w0, h0, n_window_size);
        }

        cur = ggml_add(ctx0, inpL, cur);

        ggml_set_scratch(ctx0, { 0, scr1_size, scr1, });

        struct ggml_tensor * inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF);

                // cur = mlp_ln_w*cur + mlp_ln_b
                cur = ggml_add(ctx0,
                        ggml_mul(ctx0,
                            ggml_repeat(ctx0, layer.norm2_w, cur),
                            cur),
                        ggml_repeat(ctx0, layer.norm2_b, cur));
            }

            // fully connected
            cur = ggml_mul_mat(ctx0,
                    layer.mlp_lin1_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, layer.mlp_lin1_b, cur),
                    cur);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // projection
            cur = ggml_mul_mat(ctx0,
                    layer.mlp_lin2_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, layer.mlp_lin2_b, cur),
                    cur);
        }

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    ggml_set_scratch(ctx0, { 0, scr0_size, scr0, });

    cur = ggml_cont(ctx0, ggml_permute(ctx0, inpL, 2, 0, 1, 3));

    cur = ggml_conv_2d_sk_p0(ctx0, enc.neck_conv_0, cur);

    // LayerNorm2d
    {
        // normalize along channel dimmension
        // TODO: better implementation
        cur = ggml_cont(ctx0, ggml_permute(ctx0,
                    ggml_norm(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3))),
                    2, 0, 1, 3));

        cur = ggml_add(ctx0,
                ggml_mul(ctx0,
                    ggml_repeat(ctx0, ggml_reshape_3d(ctx0, enc.neck_norm_0_w, 1, 1, n_enc_out_chans), cur),
                    cur),
                ggml_repeat(ctx0, ggml_reshape_3d(ctx0, enc.neck_norm_0_b, 1, 1, n_enc_out_chans), cur));
    }

    cur = ggml_conv_2d_s1_ph(ctx0, enc.neck_conv_1, cur);

    // LayerNorm2d
    {
        // normalize along channel dimmension
        // TODO: better implementation
        cur = ggml_cont(ctx0, ggml_permute(ctx0,
                    ggml_norm(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3))),
                    2, 0, 1, 3));

        cur = ggml_add(ctx0,
                ggml_mul(ctx0,
                    ggml_repeat(ctx0, ggml_reshape_3d(ctx0, enc.neck_norm_1_w, 1, 1, n_enc_out_chans), cur),
                    cur),
                ggml_repeat(ctx0, ggml_reshape_3d(ctx0, enc.neck_norm_1_b, 1, 1, n_enc_out_chans), cur));
    }

    // TODO: avoid copy
    cur = ggml_cpy(ctx0, cur, state.embd_img);

    ggml_set_name(cur, "check");

    ggml_set_scratch(ctx0, { 0, 0, nullptr, });

    // run the computation
    ggml_build_forward_expand(&gf, cur);
    ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

    // print
    {
        auto print_t_f32 = [&](struct ggml_tensor * t) {
            float * data = (float *)t->data;
            printf("dims: %jd %jd %jd %jd f32\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
            printf("data: ");
            for (int i = 0; i < std::min((int) t->ne[0], 10); i++) {
                printf("%f ", data[i]);
            }
            printf("\n");
            //for (int y = 0; y < 64; ++y) {
            //    for (int x = 0; x < 64; ++x) {
            //        printf("%5.2f ", data[y*64 + x]);
            //    }
            //    printf("\n");
            //}
            //printf("\n");
            for (int y = 0; y < 64; ++y) {
                for (int x = 0; x < 64; ++x) {
                    printf("%5.2f ", data[(y*64 + x)*64 + 41]);
                }
                printf("\n");
            }
            printf("\n");
            //for (int y = 0; y < 64; ++y) {
            //    for (int x = 0; x < 64; ++x) {
            //        printf("%5.2f ", data[(y*64 + x)*768 + 231]);
            //    }
            //    printf("\n");
            //}
            //printf("\n");
            double sum = 0.0;
            for (int i = 0; i < ggml_nelements(t); i++) {
                sum += data[i];
            }
            printf("sum:  %f\n", sum);
        };

        auto print_t_f16 = [&](struct ggml_tensor * t) {
            ggml_fp16_t * data = (ggml_fp16_t *)t->data;
            printf("dims: %jd %jd %jd %jd f16\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
            printf("data: ");
            for (int i = 0; i < std::min((int) t->ne[0], 10); i++) {
                printf("%f ", ggml_fp16_to_fp32(data[i]));
            }
            printf("\n");
            for (int y = 0; y < 14; ++y) {
                for (int x = 0; x < 14; ++x) {
                    printf("%7.4f ", ggml_fp16_to_fp32(data[(y*14 + x)*64 + 23]));
                }
                printf("\n");
            }
            printf("\n");
            double sum = 0.0;
            for (int i = 0; i < ggml_nelements(t); i++) {
                sum += ggml_fp16_to_fp32(data[i]);
            }
            printf("sum:  %f\n", sum);
        };

        auto * t = ggml_get_tensor(ctx0, "check");
        if (t->type == GGML_TYPE_F32) {
            print_t_f32(t);
        } else {
            print_t_f16(t);
        }
    }

    //printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);
    return true;
}

// encode a prompt
//
// - points
// - boxes
// - masks
//
// TODO: currently just encode a single point for simplicity
//
bool sam_encode_prompt(
        const sam_model     & model,
                  sam_state & state,
                        int   nx,
                        int   ny,
                  sam_point   point,
                        int   n_threads) {

    const auto & hparams = model.hparams;
    const auto & enc = model.enc_prompt;

    static size_t buf_size = 256u*1024*1024;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};

    // transform points
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py#L276
    {
        const int nmax = std::max(nx, ny);

        const float scale = hparams.n_img_size() / (float) nmax;

        const int nx_new = int(nx*scale + 0.5f);
        const int ny_new = int(ny*scale + 0.5f);

        point.x = point.x*(float(nx_new)/nx) + 0.5f;
        point.y = point.y*(float(ny_new)/ny) + 0.5f;
    }

    printf("point: %f %f\n", point.x, point.y);

    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2, 2);

    // set the input by converting the [0, 1] coordinates to [-1, 1]
    {
        float * data = (float *) inp->data;

        data[0] = 2.0f*(point.x / hparams.n_img_size()) - 1.0f;
        data[1] = 2.0f*(point.y / hparams.n_img_size()) - 1.0f;

        // padding
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L81-L85
        data[2] = 2.0f*(0.0f) - 1.0f;
        data[3] = 2.0f*(0.0f) - 1.0f;
    }

    struct ggml_tensor * cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, enc.pe)), inp);

    cur = ggml_scale(ctx0, cur, ggml_new_f32(ctx0, 2.0f*M_PI));


    // concat
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L192
    {
        struct ggml_tensor * t_sin = ggml_map_unary_f32(ctx0, cur, ggml_sam_sin);
        struct ggml_tensor * t_cos = ggml_map_unary_f32(ctx0, cur, ggml_sam_cos);

        cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, t_sin->ne[0] + t_cos->ne[0], cur->ne[1]);

        ggml_build_forward_expand(&gf, ggml_cpy(ctx0, t_sin, ggml_view_2d(ctx0, cur, t_sin->ne[0], t_sin->ne[1], cur->nb[1], 0)));
        ggml_build_forward_expand(&gf, ggml_cpy(ctx0, t_cos, ggml_view_2d(ctx0, cur, t_sin->ne[0], t_sin->ne[1], cur->nb[1], t_sin->nb[1])));

        // overwrite label == -1 with not_a_point_embed.weight
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L86
        // TODO: extend for multiple points
        ggml_build_forward_expand(&gf, ggml_cpy(ctx0, enc.not_a_pt_embd_w, ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], cur->nb[1])));

        // add point_embeddings[1] to label == 1
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L90
        ggml_build_forward_expand(&gf, ggml_add_inplace(ctx0, ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], 0), enc.pt_embd[1]));
    }

    // TODO: avoid copy
    cur = ggml_cpy(ctx0, cur, state.embd_prompt_sparse);

    ggml_build_forward_expand(&gf, cur);

    cur = ggml_repeat(ctx0,
            ggml_cont(ctx0,
                ggml_view_3d(ctx0, enc.no_mask_embd_w,
                    1, 1, enc.no_mask_embd_w->ne[0], enc.no_mask_embd_w->nb[0], enc.no_mask_embd_w->nb[0], 0)),
            state.embd_prompt_dense);

    ggml_build_forward_expand(&gf, cur);

    ggml_set_name(cur, "check");

    // run the computation
    ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

    // print
    {
        auto print_t_f32 = [&](struct ggml_tensor * t) {
            float * data = (float *)t->data;
            printf("dims: %jd %jd %jd %jd f32\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
            printf("data: ");
            for (int i = 0; i < std::min((int) t->ne[0], 10); i++) {
                printf("%f ", data[i]);
            }
            printf("\n");
            //for (int y = 0; y < 64; ++y) {
            //    for (int x = 0; x < 64; ++x) {
            //        printf("%5.2f ", data[y*64 + x]);
            //    }
            //    printf("\n");
            //}
            //printf("\n");
            for (int y = 0; y < 64; ++y) {
                for (int x = 0; x < 64; ++x) {
                    printf("%5.2f ", data[255*64*64 + y*64 + x]);
                }
                printf("\n");
            }
            printf("\n");
            //for (int y = 0; y < 64; ++y) {
            //    for (int x = 0; x < 64; ++x) {
            //        printf("%5.2f ", data[(y*64 + x)*768 + 231]);
            //    }
            //    printf("\n");
            //}
            //printf("\n");
            double sum = 0.0;
            for (int i = 0; i < ggml_nelements(t); i++) {
                sum += data[i];
            }
            printf("sum:  %f\n", sum);
        };

        auto print_t_f16 = [&](struct ggml_tensor * t) {
            ggml_fp16_t * data = (ggml_fp16_t *)t->data;
            printf("dims: %jd %jd %jd %jd f16\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
            printf("data: ");
            for (int i = 0; i < std::min((int) t->ne[0], 10); i++) {
                printf("%f ", ggml_fp16_to_fp32(data[i]));
            }
            printf("\n");
            for (int y = 0; y < 14; ++y) {
                for (int x = 0; x < 14; ++x) {
                    printf("%7.4f ", ggml_fp16_to_fp32(data[(y*14 + x)*64 + 23]));
                }
                printf("\n");
            }
            printf("\n");
            double sum = 0.0;
            for (int i = 0; i < ggml_nelements(t); i++) {
                sum += ggml_fp16_to_fp32(data[i]);
            }
            printf("sum:  %f\n", sum);
        };

        auto * t = ggml_get_tensor(ctx0, "check");
        if (t->type == GGML_TYPE_F32) {
            print_t_f32(t);
        } else {
            print_t_f16(t);
        }
    }

    //printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);
    return true;
}

bool sam_decode_mask(
        const sam_model     & model,
                  sam_state & state,
                        int   n_threads) {

    const auto & hparams = model.hparams;
    const auto & dec = model.dec;

    static size_t buf_size = 256u*1024*1024;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};
}

int main(int argc, char ** argv) {
    const int64_t t_main_start_us = ggml_time_us();

    sam_params params;
    params.model = "models/sam-vit-b/ggml-model-f16.bin";

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

#if 1
    {
        const int n = 128;
        fprintf(stderr, "%s: first %d diagonal pixels:\n", __func__, n);
        for (int i = 0; i < n; i++) {
            const int ii = i*img1.nx + i;
            fprintf(stderr, "%s:   %d: %f %f %f\n", __func__, i, img1.data[3*ii + 0], img1.data[3*ii + 1], img1.data[3*ii + 2]);
        }
    }
#endif

    int64_t t_load_us = 0;

    sam_model model;
    sam_state state;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!sam_model_load(params.model, model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    {
        static size_t buf_size = 256u*1024*1024;

        struct ggml_init_params params = {
            /*.mem_size   =*/ buf_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };

        state.ctx = ggml_init(params);

        state.embd_img = ggml_new_tensor_3d(state.ctx, GGML_TYPE_F32,
                model.hparams.n_img_embd(), model.hparams.n_img_embd(), model.hparams.n_enc_out_chans);

        // TODO: should depend on the number of points / boxes / etc
        state.embd_prompt_sparse = ggml_new_tensor_2d(state.ctx, GGML_TYPE_F32, model.hparams.n_enc_out_chans, 2);

        state.embd_prompt_dense  = ggml_new_tensor_3d(state.ctx, GGML_TYPE_F32,
                model.hparams.n_img_embd(), model.hparams.n_img_embd(), model.hparams.n_enc_out_chans);

        state.pe_img_dense = ggml_new_tensor_3d(state.ctx, GGML_TYPE_F32,
                model.hparams.n_img_embd(), model.hparams.n_img_embd(), model.hparams.n_enc_out_chans);
    }

    if (!sam_fill_dense_pe(model, state, params.n_threads)) {
        fprintf(stderr, "%s: failed to get dense positional encoding\n", __func__);
        return 1;
    }

    if (!sam_encode_image(model, state, img1, params.n_threads)) {
        fprintf(stderr, "%s: failed to encode image\n", __func__);
        return 1;
    }

    // TODO: user input
    const sam_point pt = { 414.375f, 162.796875f, };

    if (!sam_encode_prompt(model, state, img0.nx, img0.ny, pt, params.n_threads)) {
        fprintf(stderr, "%s: failed to encode prompt\n", __func__);
        return 1;
    }

    if (!sam_decode_mask(model, state, params.n_threads)) {
        fprintf(stderr, "%s: failed to decode mask\n", __func__);
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
