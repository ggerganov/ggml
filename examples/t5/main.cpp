#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// available t5 models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_SMALL,
    MODEL_BASE,
    MODEL_LARGE,
    MODEL_XL,
    MODEL_XXL,
};

static const size_t MB = 4*1024*1024;

static const std::map<e_model, size_t> MEM_REQ_MODEL = {
    { MODEL_SMALL,   74ull*MB },
    { MODEL_BASE,   142ull*MB },
    { MODEL_LARGE,  466ull*MB },
    { MODEL_XL,    1464ull*MB },
    { MODEL_XXL,   2952ull*MB },
};

static const std::map<e_model, size_t> MEM_REQ_MEMORY = {
    { MODEL_SMALL,  12ull*MB },
    { MODEL_BASE,   24ull*MB },
    { MODEL_LARGE,  70ull*MB },
    { MODEL_XL,    184ull*MB },
    { MODEL_XXL,   306ull*MB },
};

static const std::map<e_model, size_t> MEM_REQ_ENCODE = {
    { MODEL_SMALL,   80ull*MB },
    { MODEL_BASE,   128ull*MB },
    { MODEL_LARGE,  300ull*MB },
    { MODEL_XL,     680ull*MB },
    { MODEL_XXL,   1100ull*MB },
};

static const std::map<e_model, size_t> MEM_REQ_ENCODE_LAYER = {
    { MODEL_SMALL, 104ull*MB },
    { MODEL_BASE,  138ull*MB },
    { MODEL_LARGE, 208ull*MB },
    { MODEL_XL,    280ull*MB },
    { MODEL_XXL,   354ull*MB },
};

static const std::map<e_model, size_t> MEM_REQ_DECODE = {
    { MODEL_SMALL, 200ull*MB },
    { MODEL_BASE,  202ull*MB },
    { MODEL_LARGE, 204ull*MB },
    { MODEL_XL,    206ull*MB },
    { MODEL_XXL,   208ull*MB },
};

static const std::map<e_model, size_t> MEM_REQ_DECODE_LAYER = {
    { MODEL_SMALL,  32ull*MB },
    { MODEL_BASE,   44ull*MB },
    { MODEL_LARGE,  64ull*MB },
    { MODEL_XL,     84ull*MB },
    { MODEL_XXL,   110ull*MB },
};

struct t5_vocab {
    using id    = int32_t;
    using token = std::string;

    int n_vocab = 32128;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
};

// default hparams (FLAN-T5 Small)
struct t5_hparams {
    int32_t n_vocab     = 32128;
    int32_t d_ff        = 1024;
    int32_t d_kv        = 64;
    int32_t d_model     = 512;
    int32_t n_positions = 512;
    int32_t n_head      = 6;
    int32_t n_layer     = 8;
    int32_t f16         = 1;
};

struct t5_layer_encoder {
    // encoder.block.*.layer.0.SelfAttention
    struct ggml_tensor * attn_q;
    struct ggml_tensor * attn_k;
    struct ggml_tensor * attn_v;
    struct ggml_tensor * attn_o;

    // encoder.blocks.*.layer.0.layer_norm
    struct ggml_tensor * ln_0;

    // encoder.blocks.*.layer.1.DenseReluDense
    struct ggml_tensor * wi_0;
    struct ggml_tensor * wi_1;
    struct ggml_tensor * wo;

    // encoder.blocks.*.layer.1.layer_norm
    struct ggml_tensor * ln_1;
};

struct t5_layer_decoder {
    // decoder.block.*.layer.0.SelfAttention
    struct ggml_tensor * attn_q;
    struct ggml_tensor * attn_k;
    struct ggml_tensor * attn_v;
    struct ggml_tensor * attn_o;

    // decoder.blocks.*.layer.0.layer_norm
    struct ggml_tensor * ln_0;

    // decoder.blocks.*.layer.1.EncDecAttention
    struct ggml_tensor * cross_attn_q;
    struct ggml_tensor * cross_attn_k;
    struct ggml_tensor * cross_attn_v;
    struct ggml_tensor * cross_attn_o;

    // decoder.blocks.*.layer.1.layer_norm
    struct ggml_tensor * ln_1;

    // decoder.blocks.*.layer.1.DenseReluDense
    struct ggml_tensor * wi_0;
    struct ggml_tensor * wi_1;
    struct ggml_tensor * wo;

    // decoder.blocks.*.layer.1.layer_norm
    struct ggml_tensor * ln_2;
};

struct t5_model {
    e_model type = MODEL_UNKNOWN;

    t5_hparams hparams;

    // shared
    struct ggml_tensor * shared;

    // encoder.embed_tokens
    struct ggml_tensor * e_et;

    // encoder.final_layer_norm
    struct ggml_tensor * e_ln;

    // encoder.block.0.layer.0.SelfAttention.relative_attention_bias
    struct ggml_tensor * e_rab;

    // decoder.embed_tokens
    struct ggml_tensor * d_et;

    // decoder.final_layer_norm
    struct ggml_tensor * d_ln;

    // decoder.block.0.layer.0.SelfAttention.relative_attention_bias
    struct ggml_tensor * d_rab;

    // lm_head
    struct ggml_tensor * lm_head;

    std::vector<t5_layer_encoder> layers_encoder;
    std::vector<t5_layer_decoder> layers_decoder;

    // context
    struct ggml_context * ctx;
    struct ggml_context * ctx_mem;

    // tensors
    int n_loaded;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct t5_context {
    int64_t t_load_us   = 0;
    int64_t t_sample_us = 0;
    int64_t t_encode_us = 0;
    int64_t t_decode_us = 0;
    int64_t t_start_us  = 0;

    std::vector<uint8_t> buf_model;
    std::vector<uint8_t> buf_memory;
    std::vector<uint8_t> buf_compute;
    std::vector<uint8_t> buf_compute_layer;

    t5_model model;
    t5_vocab vocab;

    std::vector<float> probs;
    std::vector<float> logits;
};

template<typename T>
static void read_safe(std::ifstream& fin, T& dest) {
    fin.read((char*)& dest, sizeof(T));
}

static bool t5_model_load(const std::string & fname, t5_context & wctx) {
    fprintf(stderr, "%s: loading model from '%s'\n", __func__, fname.c_str());

    auto & model = wctx.model;
    auto & vocab = wctx.vocab;

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        read_safe(fin, magic);
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    //load hparams
    {
        auto & hparams = model.hparams;

        read_safe(fin, hparams.n_vocab);
        read_safe(fin, hparams.d_ff);
        read_safe(fin, hparams.d_kv);
        read_safe(fin, hparams.d_model);
        read_safe(fin, hparams.n_positions);
        read_safe(fin, hparams.n_head);
        read_safe(fin, hparams.n_layer);
        read_safe(fin, hparams.f16);

        assert(hparams.n_text_state == hparams.n_audio_state);

        if (hparams.n_layer == 8) {
            model.type = e_model::MODEL_SMALL;
        }

        if (hparams.n_layer == 12) {
            model.type = e_model::MODEL_BASE;
        }

        if (hparams.n_layer == 24 && hparams.n_head == 16) {
            model.type = e_model::MODEL_LARGE;
        }

        if (hparams.n_layer == 24 && hparams.n_head == 32) {
            model.type = e_model::MODEL_XL;
        }

        if (hparams.n_layer == 24 && hparams.n_head == 64) {
            model.type = e_model::MODEL_XXL;
        }

        fprintf(stderr, "%s: n_vocab       = %d\n", __func__, hparams.n_vocab);
        fprintf(stderr, "%s: d_ff          = %d\n", __func__, hparams.d_ff);
        fprintf(stderr, "%s: d_kv          = %d\n", __func__, hparams.d_kv);
        fprintf(stderr, "%s: d_model       = %d\n", __func__, hparams.d_model);
        fprintf(stderr, "%s: n_positions   = %d\n", __func__, hparams.n_positions);
        fprintf(stderr, "%s: n_head        = %d\n", __func__, hparams.n_head);
        fprintf(stderr, "%s: n_layer       = %d\n", __func__, hparams.n_layer);
        fprintf(stderr, "%s: f16           = %d\n", __func__, hparams.f16);
        fprintf(stderr, "%s: type          = %d\n", __func__, model.type);

        wctx.buf_model.resize(MEM_REQ_MODEL.at(model.type));
        wctx.buf_memory.resize(MEM_REQ_MEMORY.at(model.type));
        wctx.buf_compute.resize(std::max(MEM_REQ_ENCODE.at(model.type), MEM_REQ_DECODE.at(model.type)));
        wctx.buf_compute_layer.resize(std::max(MEM_REQ_ENCODE_LAYER.at(model.type), MEM_REQ_DECODE_LAYER.at(model.type)));
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        read_safe(fin, n_vocab);

        //if (n_vocab != model.hparams.n_vocab) {
        //    fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
        //            __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
        //    return false;
        //}

        std::string word;
        std::vector<char> tmp;

        tmp.reserve(128);

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            read_safe(fin, len);

            if (len > 0) {
                tmp.resize(len);
                fin.read(&tmp[0], tmp.size()); // read to buffer
                word.assign(&tmp[0], tmp.size());
            } else {
                // seems like we have an empty-string token in multi-language models (i = 50256)
                //fprintf(stderr, "%s: warning: empty-string token in vocab, i = %d\n", __func__, i);
                word = "";
            }

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;

            //printf("%s: vocab[%d] = '%s'\n", __func__, i, word.c_str());
        }

        vocab.n_vocab = model.hparams.n_vocab;

        if (n_vocab < model.hparams.n_vocab) {
            fprintf(stderr, "%s: adding %d extra tokens\n", __func__, model.hparams.n_vocab - n_vocab);
            for (int i = n_vocab; i < model.hparams.n_vocab; i++) {
                word = "[_extra_token_" + std::to_string(i) + "]";
                vocab.token_to_id[word] = i;
                vocab.id_to_token[i] = word;
            }
        }

        wctx.logits.reserve(vocab.n_vocab*model.hparams.d_model);
        wctx.probs.reserve(vocab.n_vocab*model.hparams.d_model);
    }

    {
        // this is the total memory required to run the inference
        const size_t mem_required =
                   wctx.buf_model.size() +
                   wctx.buf_memory.size() +
                   wctx.buf_compute.size() +
                   wctx.buf_compute_layer.size();

        fprintf(stderr, "%s: mem_required  = %7.2f MB\n", __func__, mem_required / 1024.0 / 1024.0);
    }

    // for the big tensors, we have the option to store the data in 16-bit floats
    // in order to save memory and also to speed up the computation
    const ggml_type wtype = model.hparams.f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;

    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_vocab = hparams.n_vocab;
        const int d_ff    = hparams.d_ff;
        const int d_kv    = hparams.d_kv;
        const int d_model = hparams.d_model;
        const int n_head  = hparams.n_head;
        const int n_layer = hparams.n_layer;

        ctx_size += n_vocab*d_model*ggml_type_size(wtype); // shared;
        ctx_size += n_vocab*d_model*ggml_type_size(wtype); // lm_head;

        // encoder
        {
            ctx_size += n_vocab*d_model*ggml_type_size(wtype); // e_et;
            ctx_size +=         d_model*ggml_type_size(GGML_TYPE_F32); // e_ln
            ctx_size +=       32*n_head*ggml_type_size(wtype); // e_rab
        }

        // decoder
        {
            ctx_size += n_vocab*d_model*ggml_type_size(wtype); // d_et;
            ctx_size +=         d_model*ggml_type_size(GGML_TYPE_F32); // d_ln
            ctx_size +=       32*n_head*ggml_type_size(wtype); // d_rab
        }

        // encoder layers
        {
            ctx_size += n_layer*(d_kv*n_head*d_model*ggml_type_size(wtype)); // attn_q
            ctx_size += n_layer*(d_kv*n_head*d_model*ggml_type_size(wtype)); // attn_k
            ctx_size += n_layer*(d_kv*n_head*d_model*ggml_type_size(wtype)); // attn_v
            ctx_size += n_layer*(d_kv*n_head*d_model*ggml_type_size(wtype)); // attn_o

            ctx_size += n_layer*(d_model*ggml_type_size(GGML_TYPE_F32)); // ln_0

            ctx_size += n_layer*(d_ff*d_model*ggml_type_size(wtype)); // wi_0
            ctx_size += n_layer*(d_ff*d_model*ggml_type_size(wtype)); // wi_1
            ctx_size += n_layer*(d_ff*d_model*ggml_type_size(wtype)); // wo

            ctx_size += n_layer*(d_model*ggml_type_size(GGML_TYPE_F32)); // ln_1
        }

        // decoder layers
        {
            ctx_size += n_layer*(d_kv*n_head*d_model*ggml_type_size(wtype)); // attn_q
            ctx_size += n_layer*(d_kv*n_head*d_model*ggml_type_size(wtype)); // attn_k
            ctx_size += n_layer*(d_kv*n_head*d_model*ggml_type_size(wtype)); // attn_v
            ctx_size += n_layer*(d_kv*n_head*d_model*ggml_type_size(wtype)); // attn_o

            ctx_size += n_layer*(d_model*ggml_type_size(GGML_TYPE_F32)); // ln_0

            ctx_size += n_layer*(d_kv*n_head*d_model*ggml_type_size(wtype)); // cross_attn_q
            ctx_size += n_layer*(d_kv*n_head*d_model*ggml_type_size(wtype)); // cross_attn_k
            ctx_size += n_layer*(d_kv*n_head*d_model*ggml_type_size(wtype)); // cross_attn_v
            ctx_size += n_layer*(d_kv*n_head*d_model*ggml_type_size(wtype)); // cross_attn_o

            ctx_size += n_layer*(d_model*ggml_type_size(GGML_TYPE_F32)); // ln_1

            ctx_size += n_layer*(d_ff*d_model*ggml_type_size(wtype)); // wi_0
            ctx_size += n_layer*(d_ff*d_model*ggml_type_size(wtype)); // wi_1
            ctx_size += n_layer*(d_ff*d_model*ggml_type_size(wtype)); // wo

            ctx_size += n_layer*(d_model*ggml_type_size(GGML_TYPE_F32)); // ln_2
        }

        ctx_size += (15 + 9*n_layer + 14*n_layer)*256; // object overhead

        fprintf(stderr, "%s: ggml ctx size = %7.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params;
        params.mem_size   = wctx.buf_model.size();
        params.mem_buffer = wctx.buf_model.data();

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        auto & ctx = model.ctx;

        const auto & hparams = model.hparams;

        const int n_vocab = hparams.n_vocab;
        const int d_ff    = hparams.d_ff;
        const int d_kv    = hparams.d_kv;
        const int d_model = hparams.d_model;
        const int n_head  = hparams.n_head;
        const int n_layer = hparams.n_layer;

        model.layers_encoder.resize(n_layer);
        model.layers_decoder.resize(n_layer);

        // global
        {
            model.shared  = ggml_new_tensor_2d(ctx, wtype, d_model, n_vocab);
            model.lm_head = ggml_new_tensor_2d(ctx, wtype, d_model, n_vocab);

            model.tensors["shared.weight"]  = model.shared;
            model.tensors["lm_head.weight"] = model.lm_head;
        }

        // encoder
        {
            model.e_et = ggml_new_tensor_2d(ctx,         wtype, d_model, n_vocab);
            model.e_ln = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_model);

            model.e_rab = ggml_new_tensor_2d(ctx, wtype, n_head, 32);

            // map by name
            model.tensors["encoder.embed_tokens.weight"]     = model.e_et;
            model.tensors["encoder.final_layer_norm.weight"] = model.e_ln;

            model.tensors["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = model.e_rab;

            for (int i = 0; i < n_layer; ++i) {
                auto & layer = model.layers_encoder[i];

                layer.attn_q = ggml_new_tensor_2d(ctx, wtype, d_model, d_kv*n_head);
                layer.attn_k = ggml_new_tensor_2d(ctx, wtype, d_model, d_kv*n_head);
                layer.attn_v = ggml_new_tensor_2d(ctx, wtype, d_model, d_kv*n_head);
                layer.attn_o = ggml_new_tensor_2d(ctx, wtype, d_kv*n_head, d_model);

                layer.ln_0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_model);

                layer.wi_0 = ggml_new_tensor_2d(ctx, wtype, d_model, d_ff);
                layer.wi_1 = ggml_new_tensor_2d(ctx, wtype, d_model, d_ff);
                layer.wo   = ggml_new_tensor_2d(ctx, wtype, d_ff, d_model);

                layer.ln_1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_model);

                // map by name
                model.tensors["encoder.block." + std::to_string(i) + ".layer.0.SelfAttention.q.weight"] = layer.attn_q;
                model.tensors["encoder.block." + std::to_string(i) + ".layer.0.SelfAttention.k.weight"] = layer.attn_k;
                model.tensors["encoder.block." + std::to_string(i) + ".layer.0.SelfAttention.v.weight"] = layer.attn_v;
                model.tensors["encoder.block." + std::to_string(i) + ".layer.0.SelfAttention.o.weight"] = layer.attn_o;

                model.tensors["encoder.block." + std::to_string(i) + ".layer.0.layer_norm.weight"] = layer.ln_0;

                model.tensors["encoder.block." + std::to_string(i) + ".layer.1.DenseReluDense.wi_0.weight"] = layer.wi_0;
                model.tensors["encoder.block." + std::to_string(i) + ".layer.1.DenseReluDense.wi_1.weight"] = layer.wi_1;
                model.tensors["encoder.block." + std::to_string(i) + ".layer.1.DenseReluDense.wo.weight"]   = layer.wo;

                model.tensors["encoder.block." + std::to_string(i) + ".layer.1.layer_norm.weight"] = layer.ln_1;
            }
        }

        // decoder
        {
            model.d_et = ggml_new_tensor_2d(ctx,         wtype, d_model, n_vocab);
            model.d_ln = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_model);

            model.d_rab = ggml_new_tensor_2d(ctx, wtype, n_head, 32);

            // map by name
            model.tensors["decoder.embed_tokens.weight"]     = model.d_et;
            model.tensors["decoder.final_layer_norm.weight"] = model.d_ln;

            model.tensors["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = model.d_rab;

            for (int i = 0; i < n_layer; ++i) {
                auto & layer = model.layers_decoder[i];

                layer.attn_q = ggml_new_tensor_2d(ctx, wtype, d_model, d_kv*n_head);
                layer.attn_k = ggml_new_tensor_2d(ctx, wtype, d_model, d_kv*n_head);
                layer.attn_v = ggml_new_tensor_2d(ctx, wtype, d_model, d_kv*n_head);
                layer.attn_o = ggml_new_tensor_2d(ctx, wtype, d_kv*n_head, d_model);

                layer.ln_0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_model);

                layer.cross_attn_q = ggml_new_tensor_2d(ctx, wtype, d_model, d_kv*n_head);
                layer.cross_attn_k = ggml_new_tensor_2d(ctx, wtype, d_model, d_kv*n_head);
                layer.cross_attn_v = ggml_new_tensor_2d(ctx, wtype, d_model, d_kv*n_head);
                layer.cross_attn_o = ggml_new_tensor_2d(ctx, wtype, d_kv*n_head, d_model);

                layer.ln_1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_model);

                layer.wi_0 = ggml_new_tensor_2d(ctx, wtype, d_model, d_ff);
                layer.wi_1 = ggml_new_tensor_2d(ctx, wtype, d_model, d_ff);
                layer.wo   = ggml_new_tensor_2d(ctx, wtype, d_ff, d_model);

                layer.ln_2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_model);

                // map by name
                model.tensors["decoder.block." + std::to_string(i) + ".layer.0.SelfAttention.q.weight"] = layer.attn_q;
                model.tensors["decoder.block." + std::to_string(i) + ".layer.0.SelfAttention.k.weight"] = layer.attn_k;
                model.tensors["decoder.block." + std::to_string(i) + ".layer.0.SelfAttention.v.weight"] = layer.attn_v;
                model.tensors["decoder.block." + std::to_string(i) + ".layer.0.SelfAttention.o.weight"] = layer.attn_o;

                model.tensors["decoder.block." + std::to_string(i) + ".layer.0.layer_norm.weight"] = layer.ln_0;

                model.tensors["decoder.block." + std::to_string(i) + ".layer.1.EncDecAttention.q.weight"] = layer.cross_attn_q;
                model.tensors["decoder.block." + std::to_string(i) + ".layer.1.EncDecAttention.k.weight"] = layer.cross_attn_k;
                model.tensors["decoder.block." + std::to_string(i) + ".layer.1.EncDecAttention.v.weight"] = layer.cross_attn_v;
                model.tensors["decoder.block." + std::to_string(i) + ".layer.1.EncDecAttention.o.weight"] = layer.cross_attn_o;

                model.tensors["decoder.block." + std::to_string(i) + ".layer.1.layer_norm.weight"] = layer.ln_1;

                model.tensors["decoder.block." + std::to_string(i) + ".layer.2.DenseReluDense.wi_0.weight"] = layer.wi_0;
                model.tensors["decoder.block." + std::to_string(i) + ".layer.2.DenseReluDense.wi_1.weight"] = layer.wi_1;
                model.tensors["decoder.block." + std::to_string(i) + ".layer.2.DenseReluDense.wo.weight"]   = layer.wo;

                model.tensors["decoder.block." + std::to_string(i) + ".layer.2.layer_norm.weight"] = layer.ln_2;
            }
        }
    }

    // create the ggml memory context
    {
        struct ggml_init_params params;
        params.mem_size   = wctx.buf_memory.size();
        params.mem_buffer = wctx.buf_memory.data();

        model.ctx_mem = ggml_init(params);
        if (!model.ctx_mem) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // key + value memory
    //{
    //    auto & ctx = model.ctx_mem;

    //    const auto & hparams = model.hparams;

    //    const int n_text_state = hparams.n_text_state;
    //    const int n_text_layer = hparams.n_text_layer;
    //    const int n_text_ctx   = hparams.n_text_ctx;

    //    // key/value memory for the self-attention layer
    //    {
    //        const int n_mem      = n_text_layer*n_text_ctx;
    //        const int n_elements = n_text_state*n_mem;

    //        model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);
    //        model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);
    //    }

    //    // key/value memory for the cross-attention layer
    //    {
    //        const int n_audio_ctx = hparams.n_audio_ctx;

    //        const int n_mem      = n_text_layer*n_audio_ctx;
    //        const int n_elements = n_text_state*n_mem;

    //        model.memory_cross_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);
    //        model.memory_cross_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);
    //    }

    //    const size_t memory_size =
    //        ggml_nbytes(model.memory_k)       + ggml_nbytes(model.memory_v) +
    //        ggml_nbytes(model.memory_cross_k) + ggml_nbytes(model.memory_cross_v);

    //    fprintf(stderr, "%s: memory size   = %7.2f MB\n", __func__, memory_size/1024.0/1024.0);
    //}

    // load weights
    {
        size_t total_size = 0;

        model.n_loaded = 0;

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            read_safe(fin, n_dims);
            read_safe(fin, length);
            read_safe(fin, ftype);

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[3] = { 1, 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                read_safe(fin, ne[i]);
                nelements *= ne[i];
            }

            std::string name;
            std::vector<char> tmp(length); // create a buffer
            fin.read(&tmp[0], tmp.size()); // read to buffer
            name.assign(&tmp[0], tmp.size());

            if (model.tensors.find(name) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d, %d], expected [%d, %d, %d]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], tensor->ne[2], ne[0], ne[1], ne[2]);
                return false;
            }

            const size_t bpe = (ftype == 0) ? sizeof(float) : sizeof(ggml_fp16_t);

            if (nelements*bpe != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            printf("%48s - [%5d, %5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ne[2], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
            total_size += ggml_nbytes(tensor);
            model.n_loaded++;
        }

        fprintf(stderr, "%s: model size    = %7.2f MB\n", __func__, total_size/1024.0/1024.0);

        if (model.n_loaded == 0) {
            fprintf(stderr, "%s: WARN no tensors loaded from model file - assuming empty model for testing\n", __func__);
        } else if (model.n_loaded != (int) model.tensors.size()) {
            fprintf(stderr, "%s: ERROR not all tensors loaded from model file - expected %zu, got %d\n", __func__, model.tensors.size(), model.n_loaded);
            return false;
        }
    }

    fin.close();

    return true;
}

struct t5_context * t5_init(const char * path_model) {
    ggml_time_init();

    t5_context * ctx = new t5_context;

    const int64_t t_start_us = ggml_time_us();

    ctx->t_start_us = t_start_us;

    if (!t5_model_load(path_model, *ctx)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, path_model);
        delete ctx;
        return nullptr;
    }

    ctx->t_load_us = ggml_time_us() - t_start_us;

    return ctx;
}

void t5_free(struct t5_context * ctx) {
    if (ctx) {
        if (ctx->model.ctx) {
            ggml_free(ctx->model.ctx);
        }
        if (ctx->model.ctx_mem) {
            ggml_free(ctx->model.ctx_mem);
        }
        delete ctx;
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model>\n", argv[0]);
        return -1;
    }

    const char * path_model = argv[1];

    t5_context * ctx = t5_init(path_model);
    if (!ctx) {
        fprintf(stderr, "%s: failed to initialize T5 context\n", __func__);
        return -1;
    }

    fprintf(stderr, "%s: model loaded in %7.2f ms\n", __func__, ctx->t_load_us/1000.0);

    t5_free(ctx);

    return 0;
}
