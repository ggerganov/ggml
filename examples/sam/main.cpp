#include "ggml.h"

#include "common.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
    int32_t n_dec_heads       = 8;
    int32_t ftype             = 1;
    float   iou_threshold     = 0.88f;

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
    struct ggml_tensor * embd_prompt_sparse;
    struct ggml_tensor * embd_prompt_dense;
    struct ggml_tensor * pe_img_dense;

    struct ggml_tensor * low_res_masks;
    struct ggml_tensor * iou_predictions;

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
                ctx_size += n_hypernet_mpls_count*2*n_enc_out_chans*n_enc_out_chans*ggml_type_sizef(GGML_TYPE_F16);
                ctx_size += n_hypernet_mpls_count*2*n_enc_out_chans*                ggml_type_sizef(GGML_TYPE_F32);
                ctx_size += n_hypernet_mpls_count*n_enc_out_chans*(n_img_embd/2)*ggml_type_sizef(GGML_TYPE_F16);
                ctx_size += n_hypernet_mpls_count*(n_img_embd/2)*                ggml_type_sizef(GGML_TYPE_F32);

                // iou_prediction_head
                ctx_size += 2*n_enc_out_chans*n_enc_out_chans*ggml_type_sizef(GGML_TYPE_F16);
                ctx_size += 2*n_enc_out_chans*                ggml_type_sizef(GGML_TYPE_F32);
                ctx_size += n_pt_embd*n_enc_out_chans*ggml_type_sizef(GGML_TYPE_F16);
                ctx_size += n_pt_embd*                ggml_type_sizef(GGML_TYPE_F32);

                // iou_token_w
                ctx_size += n_enc_out_chans*ggml_type_sizef(GGML_TYPE_F32);

                // mask_tokens_w
                ctx_size += n_pt_embd*n_enc_out_chans*ggml_type_sizef(GGML_TYPE_F32);
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

        if (n_tensors != model.tensors.size()) {
            fprintf(stderr, "%s: model file has %d tensors, but %d tensors were expected\n", __func__, n_tensors, (int) model.tensors.size());
            return false;
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
        // auto print_t_f32 = [&](struct ggml_tensor * t) {
        //     float * data = (float *)t->data;
        //     printf("dims: %jd %jd %jd %jd f32\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
        //     printf("data: ");
        //     for (int i = 0; i < std::min((int) t->ne[0], 10); i++) {
        //         printf("%f ", data[i]);
        //     }
        //     printf("\n");
        //     //for (int y = 0; y < 64; ++y) {
        //     //    for (int x = 0; x < 64; ++x) {
        //     //        printf("%5.2f ", data[y*64 + x]);
        //     //    }
        //     //    printf("\n");
        //     //}
        //     //printf("\n");
        //     for (int y = 0; y < 64; ++y) {
        //         for (int x = 0; x < 64; ++x) {
        //             printf("%5.2f ", data[(y*64 + x)*64 + 41]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        //     //for (int y = 0; y < 64; ++y) {
        //     //    for (int x = 0; x < 64; ++x) {
        //     //        printf("%5.2f ", data[(y*64 + x)*768 + 231]);
        //     //    }
        //     //    printf("\n");
        //     //}
        //     //printf("\n");
        //     double sum = 0.0;
        //     for (int i = 0; i < ggml_nelements(t); i++) {
        //         sum += data[i];
        //     }
        //     printf("sum:  %f\n", sum);
        // };

        // auto print_t_f16 = [&](struct ggml_tensor * t) {
        //     ggml_fp16_t * data = (ggml_fp16_t *)t->data;
        //     printf("dims: %jd %jd %jd %jd f16\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
        //     printf("data: ");
        //     for (int i = 0; i < std::min((int) t->ne[0], 10); i++) {
        //         printf("%f ", ggml_fp16_to_fp32(data[i]));
        //     }
        //     printf("\n");
        //     for (int y = 0; y < 14; ++y) {
        //         for (int x = 0; x < 14; ++x) {
        //             printf("%7.4f ", ggml_fp16_to_fp32(data[(y*14 + x)*64 + 23]));
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        //     double sum = 0.0;
        //     for (int i = 0; i < ggml_nelements(t); i++) {
        //         sum += ggml_fp16_to_fp32(data[i]);
        //     }
        //     printf("sum:  %f\n", sum);
        // };

        // auto * t = ggml_get_tensor(ctx0, "check");
        // if (t->type == GGML_TYPE_F32) {
        //     print_t_f32(t);
        // } else {
        //     print_t_f16(t);
        // }
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

    // TODO: avoid copy
    cur = ggml_cpy(ctx0, cur, state.embd_prompt_dense);

    ggml_build_forward_expand(&gf, cur);

    ggml_set_name(cur, "check");

    // run the computation
    ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

    // print
    {
        // auto print_t_f32 = [&](struct ggml_tensor * t) {
        //     float * data = (float *)t->data;
        //     printf("dims: %jd %jd %jd %jd f32\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
        //     printf("data: ");
        //     for (int i = 0; i < std::min((int) t->ne[0], 256); i++) {
        //         printf("%f ", data[i]);
        //     }
        //     printf("\n");
        //     //for (int y = 0; y < 64; ++y) {
        //     //    for (int x = 0; x < 64; ++x) {
        //     //        printf("%5.2f ", data[y*64 + x]);
        //     //    }
        //     //    printf("\n");
        //     //}
        //     //printf("\n");
        //     // for (int y = 0; y < 64; ++y) {
        //     //     for (int x = 0; x < 64; ++x) {
        //     //         printf("%5.2f ", data[255*64*64 + y*64 + x]);
        //     //     }
        //     //     printf("\n");
        //     // }
        //     // printf("\n");
        //     //for (int y = 0; y < 64; ++y) {
        //     //    for (int x = 0; x < 64; ++x) {
        //     //        printf("%5.2f ", data[(y*64 + x)*768 + 231]);
        //     //    }
        //     //    printf("\n");
        //     //}
        //     //printf("\n");
        //     double sum = 0.0;
        //     for (int i = 0; i < ggml_nelements(t); i++) {
        //         sum += data[i];
        //     }
        //     printf("sum:  %f\n", sum);
        // };

        // auto print_t_f16 = [&](struct ggml_tensor * t) {
        //     ggml_fp16_t * data = (ggml_fp16_t *)t->data;
        //     printf("dims: %jd %jd %jd %jd f16\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
        //     printf("data: ");
        //     for (int i = 0; i < std::min((int) t->ne[0], 256); i++) {
        //         printf("%f ", ggml_fp16_to_fp32(data[i]));
        //     }
        //     printf("\n");
        //     for (int y = 0; y < 14; ++y) {
        //         for (int x = 0; x < 14; ++x) {
        //             printf("%7.4f ", ggml_fp16_to_fp32(data[(y*14 + x)*64 + 23]));
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        //     double sum = 0.0;
        //     for (int i = 0; i < ggml_nelements(t); i++) {
        //         sum += ggml_fp16_to_fp32(data[i]);
        //     }
        //     printf("sum:  %f\n", sum);
        // };

        // auto * t = ggml_get_tensor(ctx0, "check");
        // if (t->type == GGML_TYPE_F32) {
        //     print_t_f32(t);
        // } else {
        //     print_t_f16(t);
        // }
    }

    //printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);
    return true;
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
    Qcur = ggml_add(ctx0,
            ggml_repeat(ctx0, attn.q_b, Qcur),
            Qcur);

    Kcur = ggml_mul_mat(ctx0, attn.k_w, keys);
    Kcur = ggml_add(ctx0,
            ggml_repeat(ctx0, attn.k_b, Kcur),
            Kcur);

    Vcur = ggml_mul_mat(ctx0, attn.v_w, values);
    Vcur = ggml_add(ctx0,
            ggml_repeat(ctx0, attn.v_b, Vcur),
            Vcur);

    // TODO: use stratch memory
    // ggml_set_scratch(ctx0, { 0, scr0_size, scr0, });

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

    struct ggml_tensor * KQ_scaled =
        ggml_scale_inplace(ctx0,
                KQ,
                ggml_new_f32(ctx0, 1.0f/sqrt(float(Q->ne[0]))));

    struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_scaled);

    struct ggml_tensor * KQV = ggml_mul_mat(ctx0, KQ_soft_max, ggml_cont(ctx0, ggml_transpose(ctx0, V)));

    struct ggml_tensor * KQV_merged = ggml_cont(ctx0, ggml_transpose(ctx0, KQV));
    KQV_merged = ggml_cont(ctx0, ggml_permute(ctx0, KQV_merged, 0, 2, 1, 3));
    KQV_merged = ggml_cont(ctx0, ggml_reshape_3d(ctx0, KQV_merged, KQV_merged->ne[0]*KQV_merged->ne[1], KQV_merged->ne[2], KQV_merged->ne[3]));
    KQV_merged = ggml_mul_mat(ctx0, attn.out_w, KQV_merged);
    KQV_merged = ggml_add(ctx0,
                ggml_repeat(ctx0, attn.out_b, KQV_merged),
                KQV_merged);

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
    cur = ggml_add(ctx0,
            ggml_repeat(ctx0, b_0, cur),
            cur);
    cur = ggml_relu(ctx0, cur);

    cur = ggml_mul_mat(ctx0, w_1, cur);
    cur = ggml_add(ctx0,
            ggml_repeat(ctx0, b_1, cur),
            cur);
    cur = ggml_relu(ctx0, cur);

    cur = ggml_mul_mat(ctx0, w_2, cur);
    cur = ggml_add(ctx0,
            ggml_repeat(ctx0, b_2, cur),
            cur);

    return cur;
}

bool sam_decode_mask(
        const sam_model     & model,
                  sam_state & state,
                        int   n_threads) {

    const auto & hparams = model.hparams;
    const auto & dec = model.dec;
    const int n_img_embd = hparams.n_img_embd();

    static size_t buf_size = 384u*1024*1024;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};

    auto print_t_f32 = [&](const char* title, struct ggml_tensor * t) {
        printf("%s\n", title);
        float * data = (float *)t->data;
        printf("dims: %jd %jd %jd %jd f32\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
        printf("First 10 elements:\n");
        for (int i = 0; i < std::min((int) t->ne[0], 10); i++) {
            printf("%f ", data[i]);
        }
        printf("\n");
        double sum = 0.0;
        for (int i = 0; i < ggml_nelements(t); i++) {
            sum += data[i];
        }
        printf("sum:  %f\n\n", sum);
    };
    print_t_f32("embd_img", state.embd_img);
    print_t_f32("embd_prompt_dense", state.embd_prompt_dense);
    print_t_f32("embd_prompt_sparse", state.embd_prompt_sparse);
    print_t_f32("pe_img_dense", state.pe_img_dense);

    struct ggml_tensor * tokens = {};
    {
        // Concatenate output tokens
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L120
        const auto& sparse = state.embd_prompt_sparse;

        tokens = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, dec.iou_token_w->ne[0], dec.iou_token_w->ne[1] + dec.mask_tokens_w->ne[1] + sparse->ne[1], sparse->ne[2]);

        const size_t offsets[3] = { 0, dec.iou_token_w->ne[1]*tokens->nb[1], dec.iou_token_w->ne[1]*tokens->nb[1] + dec.mask_tokens_w->ne[1]*tokens->nb[1] };
        ggml_build_forward_expand(&gf, ggml_cpy(ctx0, dec.iou_token_w,   ggml_view_2d(ctx0, tokens, tokens->ne[0], dec.iou_token_w->ne[1],   tokens->nb[1], offsets[0])));
        ggml_build_forward_expand(&gf, ggml_cpy(ctx0, dec.mask_tokens_w, ggml_view_2d(ctx0, tokens, tokens->ne[0], dec.mask_tokens_w->ne[1], tokens->nb[1], offsets[1])));
        ggml_build_forward_expand(&gf, ggml_cpy(ctx0, sparse,            ggml_view_2d(ctx0, tokens, tokens->ne[0], sparse->ne[1],            tokens->nb[1], offsets[2])));
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
            state.embd_prompt_dense);

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
        ggml_build_forward_expand(&gf, src);

        pos_src = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, state.pe_img_dense->ne[0], state.pe_img_dense->ne[1], state.pe_img_dense->ne[2], tokens->ne[2]);
        pos_src = ggml_repeat(ctx0,
            state.pe_img_dense,
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
        ggml_build_forward_expand(&gf, pos_src);
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

            queries = ggml_norm(ctx0, queries);
            queries = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, tfm_layer.norm1_w, queries),
                        queries),
                    ggml_repeat(ctx0, tfm_layer.norm1_b, queries));

            // Cross attention block, tokens attending to image embedding
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L163
            struct ggml_tensor * q_1 = ggml_add(ctx0, queries, tokens);
            struct ggml_tensor * k_1 = ggml_add(ctx0, keys, pos_src);

            struct ggml_tensor * cross_attn_token_to_img = sam_decode_mask_transformer_attn(tfm_layer.cross_attn_token_to_img, q_1, k_1, keys, ctx0, model);

            queries = ggml_add(ctx0, queries, cross_attn_token_to_img);
            queries = ggml_norm(ctx0, queries);
            queries = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, tfm_layer.norm2_w, queries),
                        queries),
                    ggml_repeat(ctx0, tfm_layer.norm2_b, queries));

            // MLP block
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L170
            struct ggml_tensor * mlp_out = ggml_mul_mat(ctx0,
                tfm_layer.mlp_lin1_w,
                queries);

            mlp_out = ggml_add(ctx0,
                    ggml_repeat(ctx0, tfm_layer.mlp_lin1_b, mlp_out),
                    mlp_out);

            // RELU activation
            mlp_out = ggml_relu(ctx0, mlp_out);
            mlp_out = ggml_mul_mat(ctx0,
                    tfm_layer.mlp_lin2_w,
                    mlp_out);

            mlp_out = ggml_add(ctx0,
                    ggml_repeat(ctx0, tfm_layer.mlp_lin2_b, mlp_out),
                    mlp_out);

            queries = ggml_add(ctx0, queries, mlp_out);
            queries = ggml_norm(ctx0, queries);
            queries = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, tfm_layer.norm3_w, queries),
                        queries),
                    ggml_repeat(ctx0, tfm_layer.norm3_b, queries));

            // Cross attention block, image embedding attending to tokens
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L175
            struct ggml_tensor * q_2 = ggml_add(ctx0, queries, tokens);
            struct ggml_tensor * k_2 = ggml_add(ctx0, keys, pos_src);

            struct ggml_tensor * cross_attn_img_to_token = sam_decode_mask_transformer_attn(tfm_layer.cross_attn_img_to_token, k_2, q_2, queries, ctx0, model);
            keys = ggml_add(ctx0, keys, cross_attn_img_to_token);
            keys = ggml_norm(ctx0, keys);
            keys = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, tfm_layer.norm4_w, keys),
                        keys),
                    ggml_repeat(ctx0, tfm_layer.norm4_b, keys));
        }

        // Apply the final attention layer from the points to the image
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L99
        struct ggml_tensor * q = ggml_add(ctx0, queries, tokens);
        struct ggml_tensor * k = ggml_add(ctx0, keys, pos_src);

        struct ggml_tensor * final_attn_token_to_img = sam_decode_mask_transformer_attn(dec.transformer_final_attn_token_to_img, q, k, keys, ctx0, model);

        queries = ggml_add(ctx0, queries, final_attn_token_to_img);
        queries = ggml_norm(ctx0, queries);
        queries = ggml_add(ctx0,
                ggml_mul(ctx0,
                    ggml_repeat(ctx0, dec.transformer_norm_final_w, queries),
                    queries),
                ggml_repeat(ctx0, dec.transformer_norm_final_b, queries));
    }

    struct ggml_tensor * iou_pred = ggml_view_2d(ctx0, queries, queries->ne[0], queries->ne[2], queries->nb[2], 0);
    const int num_mask_tokens = 4; // num_multimask_outputs + 1
    struct ggml_tensor * mask_tokens_out = ggml_view_3d(ctx0, queries, queries->ne[0], num_mask_tokens, queries->ne[2], queries->nb[1], num_mask_tokens*queries->nb[1], queries->nb[1]);

    // Upscale mask embeddings and predict masks using the mask tokens
    // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L136
    keys = ggml_cont(ctx0, ggml_transpose(ctx0, keys));
    keys = ggml_view_4d(ctx0, keys, srcNE[0], srcNE[1], srcNE[2], srcNE[3], srcNE[0]*keys->nb[0], keys->nb[1], keys->nb[2], 0);

    struct ggml_tensor * upscaled_embedding = {};
    {
        // ConvTranspose2d
        keys = ggml_conv_transpose_2d_p0(ctx0, dec.output_upscaling_0_w, keys, 2);
        keys = ggml_add(ctx0, ggml_repeat(ctx0, dec.output_upscaling_0_b, keys), keys);

        // LayerNorm2d
        {
            // normalize along channel dimmension
            // TODO: better implementation
            keys = ggml_cont(ctx0, ggml_permute(ctx0,
                        ggml_norm(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, keys, 1, 2, 0, 3))),
                        2, 0, 1, 3));

            keys = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, ggml_reshape_3d(ctx0, dec.output_upscaling_1_w, 1, 1, n_img_embd), keys),
                        keys),
                    ggml_repeat(ctx0, ggml_reshape_3d(ctx0, dec.output_upscaling_1_b, 1, 1, n_img_embd), keys));
        }

        // GELU activation
        keys = ggml_gelu(ctx0, keys);

        // ConvTranspose2d
        keys = ggml_conv_transpose_2d_p0(ctx0, dec.output_upscaling_3_w, keys, 2);
        keys = ggml_add(ctx0, ggml_repeat(ctx0, dec.output_upscaling_3_b, keys), keys);

        // GELU activation
        keys = ggml_gelu(ctx0, keys);
        upscaled_embedding = ggml_reshape_3d(ctx0, keys, keys->ne[0]*keys->ne[1], keys->ne[2], keys->ne[3]);
        upscaled_embedding = ggml_cont(ctx0, ggml_transpose(ctx0, upscaled_embedding)); // TODO: Shouldn't be needed
    }

    struct ggml_tensor * hyper_in = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_img_embd/2, num_mask_tokens, mask_tokens_out->ne[2]);
    for (int i = 0; i < num_mask_tokens; ++i) {
        const auto& mlp = dec.output_hypernet_mlps[i];
        struct ggml_tensor * in = ggml_view_2d(ctx0, mask_tokens_out, mask_tokens_out->ne[0], mask_tokens_out->ne[2], mask_tokens_out->nb[1], i*mask_tokens_out->nb[1]);
        struct ggml_tensor * out = sam_decode_mask_mlp_relu_3(in, mlp.w_0, mlp.b_0, mlp.w_1, mlp.b_1, mlp.w_2, mlp.b_2, ctx0);
        ggml_build_forward_expand(&gf, ggml_cpy(ctx0, out, ggml_view_2d(ctx0, hyper_in, hyper_in->ne[0], hyper_in->ne[2], hyper_in->nb[1], i*hyper_in->nb[1])));
    }

    struct ggml_tensor * masks = ggml_mul_mat(ctx0, hyper_in, upscaled_embedding);
    masks = ggml_cont(ctx0, ggml_transpose(ctx0, masks)); // TODO: Shouldn't be needed
    masks = ggml_cont(ctx0, ggml_reshape_4d(ctx0, masks, keys->ne[0], keys->ne[1], masks->ne[1], keys->ne[3]));

    // Generate mask quality predictions
    // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L146
    state.iou_predictions = sam_decode_mask_mlp_relu_3(iou_pred, dec.iou_prediction_head_0_w, dec.iou_prediction_head_0_b, dec.iou_prediction_head_1_w, dec.iou_prediction_head_1_b, dec.iou_prediction_head_2_w, dec.iou_prediction_head_2_b, ctx0);

    // Select the correct mask or masks for output
    // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L101
    state.iou_predictions = ggml_view_1d(ctx0, state.iou_predictions, state.iou_predictions->ne[0] - 1, state.iou_predictions->nb[0]);

    state.low_res_masks = ggml_view_4d(ctx0, masks, masks->ne[0], masks->ne[1], masks->ne[2] - 1, masks->ne[3],
                                                    masks->nb[0], masks->nb[1], masks->nb[2],
                                                    masks->nb[2] /* offset*/);
    ggml_set_name(queries, "queries");
    ggml_set_name(upscaled_embedding, "upscaled_embedding");
    ggml_set_name(state.low_res_masks, "low_res_masks");
    ggml_set_name(state.iou_predictions, "iou_predictions");
    ggml_set_name(hyper_in, "hyper_in");

    ggml_build_forward_expand(&gf, state.iou_predictions);
    ggml_build_forward_expand(&gf, state.low_res_masks);

    // run the computation
    ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

    auto * t = ggml_get_tensor(ctx0, "queries");
    print_t_f32("queries", t);
    t = ggml_get_tensor(ctx0, "upscaled_embedding");
    print_t_f32("upscaled_embedding", t);
    t = ggml_get_tensor(ctx0, "low_res_masks");
    print_t_f32("low_res_masks", t);
    t = ggml_get_tensor(ctx0, "iou_predictions");
    print_t_f32("iou_predictions", t);
    t = ggml_get_tensor(ctx0, "hyper_in");
    print_t_f32("hyper_in", t);

    ggml_free(ctx0);
    return true;
}

bool sam_write_masks(const sam_hparams& hparams, int nx, int ny, const sam_state & state) {
    if (state.low_res_masks->ne[2] == 0) return true;
    if (state.low_res_masks->ne[2] != state.iou_predictions->ne[0]) {
        printf("Error: number of masks (%jd) does not match number of iou predictions (%jd)\n", state.low_res_masks->ne[2], state.iou_predictions->ne[0]);
        return false;
    }

    const int n_img_size = hparams.n_img_size();
    const float iou_threshold = hparams.iou_threshold;

    const int ne0 = state.low_res_masks->ne[0];
    const int ne1 = state.low_res_masks->ne[1];
    const int ne2 = state.low_res_masks->ne[2];

    // TODO: Calculate and filter based on stability score and crop boxes

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

        sam_image_u8 res;
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

                    if (v > 0) {
                        const uint8_t v2 = std::min(std::max((int)std::round(v*255.f), 0), 255);
                        res.data[iy*nx + ix] = v2;
                    }
                }
            }
        }

        std::string filename = "mask_out_" + std::to_string(i) + ".png";
        if (!stbi_write_png(filename.c_str(), res.nx, res.ny, 1, res.data.data(), res.nx)) {
            printf("%s: failed to write mask %s\n", __func__, filename.c_str());
            return false;
        }
    }


    return true;
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

    if (!sam_write_masks(model.hparams, img0.nx, img0.ny, state)) {
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
