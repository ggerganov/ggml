#include "ggml/ggml.h"

#include "common.h"
#include "common-ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#if !defined(_WIN32)
#define DOLLY_INTERACTIVE_PORT
#endif

#if defined(DOLLY_INTERACTIVE_PORT)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// default hparams (Dolly-V2 3B)
struct dollyv2_hparams {
    int32_t n_vocab = 50254; // tokenizer.vocab_size
    int32_t n_ctx   = 2048;  // model.config.max_position_embeddings
    int32_t n_embd  = 2560;  // model.config.hidden_size
    int32_t n_head  = 32;    // model.config.num_attention_heads
    int32_t n_layer = 32;    // model.config.num_hidden_layers
    int32_t n_rot   = 20;    // rotary_pct[25%] * (n_embd / n_head)
    int32_t par_res = 1; // 1 = true, 0 = false
    int32_t ftype   = GGML_FTYPE_MOSTLY_F16;
    float   eps     = 1e-5f;
};

const std::string INSTRUCTION_KEY = "### Instruction:";
const std::string RESPONSE_KEY    = "### Response:";
const std::string END_KEY         = "### End";
const std::string INTRO_BLURB     = "Below is an instruction that describes a task. Write a response that appropriately completes the request.";

// dollyv2 prompt format
std::string prompt_for_generation(const std::string& instruction) {
    return INTRO_BLURB + "\n\n" + INSTRUCTION_KEY + "\n" + instruction + "\n\n" + RESPONSE_KEY + "\n";
}

struct dollyv2_layer {
    // pre normalization
    struct ggml_tensor * ln_1_g;
    struct ggml_tensor * ln_1_b;

    // attention
    struct ggml_tensor * c_attn_attn_w;
    struct ggml_tensor * c_attn_attn_b;

    struct ggml_tensor * c_attn_proj_w;
    struct ggml_tensor * c_attn_proj_b;

    // post normalization
    struct ggml_tensor * ln_2_g;
    struct ggml_tensor * ln_2_b;

    // ff
    struct ggml_tensor * c_mlp_fc_w;
    struct ggml_tensor * c_mlp_fc_b;

    struct ggml_tensor * c_mlp_proj_w;
    struct ggml_tensor * c_mlp_proj_b;
};

struct dollyv2_model {
    dollyv2_hparams hparams;

    // normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * wte; // position embedding

    struct ggml_tensor * lmh_g; // language model head
    //struct ggml_tensor * lmh_b; // language model bias

    std::vector<dollyv2_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// load the model's weights from a file
bool dollyv2_model_load(const std::string & fname, dollyv2_model & model, gpt_vocab & vocab) {
    printf("%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // load hparams
    {
        auto & hparams = model.hparams;

        fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        fin.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fin.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fin.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *) &hparams.n_rot,   sizeof(hparams.n_rot));
        fin.read((char *) &hparams.par_res, sizeof(hparams.par_res));
        fin.read((char *) &hparams.ftype,   sizeof(hparams.ftype));

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: n_rot   = %d\n", __func__, hparams.n_rot);
        printf("%s: par_res = %d\n", __func__, hparams.par_res);
        printf("%s: ftype   = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr   = %d\n", __func__, qntvr);

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }

    // load vocab
    {
        const int32_t n_vocab = model.hparams.n_vocab;

        std::string word;
        std::vector<char> buf(128);

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            buf.resize(len);
            fin.read((char *) buf.data(), len);
            word.assign(buf.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }

        vocab.add_special_token("### End");
        vocab.add_special_token("### Instruction:");
        vocab.add_special_token("### Response:");
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

    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // ln_f_g
        ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // ln_f_b

        ctx_size += n_embd*n_vocab*ggml_type_sizef(wtype); // wte

        ctx_size += n_embd*n_vocab*ggml_type_sizef(wtype);           // lmh_g
        //ctx_size +=        n_vocab*ggml_type_sizef(GGML_TYPE_F32); // lmh_b

        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_1_g
        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_1_b

        ctx_size += n_layer*(3*n_embd*n_embd*ggml_type_sizef(wtype));         // c_attn_attn_w
        ctx_size += n_layer*(       3*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_attn_attn_b

        ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype));         // c_attn_proj_w
        ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_attn_proj_b

        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_2_g
        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_2_b

        ctx_size += n_layer*(4*n_embd*n_embd*ggml_type_sizef(wtype));         // c_mlp_fc_w
        ctx_size += n_layer*(       4*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_mlp_fc_b

        ctx_size += n_layer*(4*n_embd*n_embd*ggml_type_sizef(wtype));         // c_mlp_proj_w
        ctx_size += n_layer*(         n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_mlp_proj_b

        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_k
        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_v

        ctx_size += (6 + 16*n_layer)*512; // object overhead

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.wte    = ggml_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);

        model.ln_f_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.ln_f_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        model.lmh_g  = ggml_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);
        //model.lmh_b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_vocab);

        // map by name
        model.tensors["gpt_neox.embed_in.weight"] = model.wte;

        model.tensors["gpt_neox.final_layer_norm.weight"] = model.ln_f_g;
        model.tensors["gpt_neox.final_layer_norm.bias"]   = model.ln_f_b;

        model.tensors["embed_out.weight"] = model.lmh_g;
        //model.tensors["lm_head.bias"]   = model.lmh_b;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            layer.ln_1_g          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.ln_1_b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.c_attn_attn_w   = ggml_new_tensor_2d(ctx, wtype,           n_embd, 3*n_embd);
            layer.c_attn_attn_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3*n_embd);

            layer.c_attn_proj_w   = ggml_new_tensor_2d(ctx, wtype,           n_embd,   n_embd);
            layer.c_attn_proj_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.ln_2_g          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.ln_2_b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.c_mlp_fc_w      = ggml_new_tensor_2d(ctx, wtype,           n_embd, 4*n_embd);
            layer.c_mlp_fc_b      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*n_embd);

            layer.c_mlp_proj_w    = ggml_new_tensor_2d(ctx, wtype,         4*n_embd,   n_embd);
            layer.c_mlp_proj_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            // map by name

            // unmapped: attention.rotary_emb, mlp.act

            model.tensors["gpt_neox.layers." + std::to_string(i) + ".input_layernorm.weight"] = layer.ln_1_g;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".input_layernorm.bias"]   = layer.ln_1_b;

            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.query_key_value.weight"] = layer.c_attn_attn_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.query_key_value.bias"]   = layer.c_attn_attn_b;

            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.dense.weight"] = layer.c_attn_proj_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.dense.bias"]   = layer.c_attn_proj_b;

            model.tensors["gpt_neox.layers." + std::to_string(i) + ".post_attention_layernorm.weight"] = layer.ln_2_g;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".post_attention_layernorm.bias"]   = layer.ln_2_b;

            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_h_to_4h.weight"] = layer.c_mlp_fc_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_h_to_4h.bias"]   = layer.c_mlp_fc_b;

            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_4h_to_h.weight"] = layer.c_mlp_proj_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_4h_to_h.bias"]   = layer.c_mlp_proj_b;
        }
    }

    // key + value memory
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;

        const int64_t n_mem      = n_layer*n_ctx;
        const int64_t n_elements = n_embd*n_mem;

        model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);
        model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);

        const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

        printf("%s: memory_size = %8.2f MB, n_mem = %" PRId64 "\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }

    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;

        printf("%s: ", __func__);

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.c_str());
                return false;
            }

            auto tensor = model.tensors[name];
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.c_str());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%5d, %5d], expected [%5d, %5d]\n",
                        __func__, name.c_str(), (int) tensor->ne[0], (int) tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            // for debugging
            if (0) {
                printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", name.c_str(), ne[0], ne[1], ggml_type_name(ggml_type(ttype)), ggml_nbytes(tensor)/1024.0/1024.0, ggml_nbytes(tensor));
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.c_str(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 8 == 0) {
                printf(".");
                fflush(stdout);
            }
        }

        printf(" done\n");

        printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size/1024.0/1024.0, n_tensors);
    }

    fin.close();

    return true;
}

// feed-forward network
ggml_tensor * gpt_neox_ff(
        const dollyv2_layer & layer,
        ggml_context        * ctx0,
        ggml_tensor         * inp,
        float                 eps) {
    ggml_tensor * cur = ggml_norm(ctx0, inp, eps);

    cur = ggml_add(ctx0,
        ggml_mul(ctx0,
            ggml_repeat(ctx0, layer.ln_2_g, cur),
            cur),
        ggml_repeat(ctx0, layer.ln_2_b, cur));

    cur = ggml_mul_mat(ctx0,
            layer.c_mlp_fc_w,
            cur);

    cur = ggml_add(ctx0,
            ggml_repeat(ctx0, layer.c_mlp_fc_b, cur),
            cur);

    // GELU activation
    cur = ggml_gelu(ctx0, cur);

    // projection
    // cur = proj_w*cur + proj_b
    cur = ggml_mul_mat(ctx0,
            layer.c_mlp_proj_w,
            cur);

    cur = ggml_add(ctx0,
            ggml_repeat(ctx0, layer.c_mlp_proj_b, cur),
            cur);
    return cur;
}

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
bool dollyv2_eval(
        const dollyv2_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w,
              size_t                     & mem_per_token) {
    const int N = embd_inp.size();

    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_vocab;
    const int n_rot   = hparams.n_rot;

    static size_t buf_size = 256u*1024*1024;
    static void * buf = malloc(buf_size);

    if (mem_per_token > 0 && mem_per_token*N > buf_size) {
        const size_t buf_size_new = 1.1*(mem_per_token*N); // add 10% to account for ggml object overhead
        //printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

        // reallocate
        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr) {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return false;
        }
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = { };

    // KQ_pos - contains the positions
    struct ggml_tensor * KQ_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    int * data = (int *) KQ_pos->data;
    for (int i = 0; i < N; ++i) {
        data[i] = n_past + i;
    }

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, embd_inp.data(), N*ggml_element_size(embd));

    // wte
    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.wte, embd);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur;

        // self-attention
        {
            {
                cur = ggml_norm(ctx0, inpL, hparams.eps);

                cur = ggml_add(ctx0,
                        ggml_mul(ctx0,
                            ggml_repeat(ctx0, model.layers[il].ln_1_g, cur),
                            cur),
                        ggml_repeat(ctx0, model.layers[il].ln_1_b, cur));
            }

            // compute QKV
            {
                cur = ggml_mul_mat(ctx0,
                        model.layers[il].c_attn_attn_w,
                        cur);

                cur = ggml_add(ctx0,
                        ggml_repeat(ctx0, model.layers[il].c_attn_attn_b, cur),
                        cur);
            }

            struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd/n_head, n_head, N, cur->nb[1]/n_head, cur->nb[1], 0*sizeof(float)*n_embd/n_head));
            struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd/n_head, n_head, N, cur->nb[1]/n_head, cur->nb[1], 1*sizeof(float)*n_embd/n_head));
            struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd/n_head, n_head, N, cur->nb[1]/n_head, cur->nb[1], 2*sizeof(float)*n_embd/n_head));

            // using mode = 2 for GPT-NeoX mode
            Qcur = ggml_rope_inplace(ctx0, Qcur, KQ_pos, n_rot, 2, 0);
            Kcur = ggml_rope_inplace(ctx0, Kcur, KQ_pos, n_rot, 2, 0);

            // store key and value to memory
            {
                Vcur = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcur, n_embd, N));

                struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_2d(ctx0, model.memory_v, N, n_embd,
                        (   n_ctx)*ggml_element_size(model.memory_v),
                        (il*n_ctx)*ggml_element_size(model.memory_v)*n_embd + n_past*ggml_element_size(model.memory_v));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        Qcur,
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, model.memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_k)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor * KQ_scaled =
                ggml_scale_inplace(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
                        );

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_tensor * V =
                ggml_view_3d(ctx0, model.memory_v,
                        n_past + N, n_embd/n_head, n_head,
                        n_ctx*ggml_element_size(model.memory_v),
                        n_ctx*ggml_element_size(model.memory_v)*n_embd/n_head,
                        il*n_ctx*ggml_element_size(model.memory_v)*n_embd);

            // KQV = transpose(V) * KQ_soft_max
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection
            {
                cur = ggml_mul_mat(ctx0,
                        model.layers[il].c_attn_proj_w,
                        cur);

                cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].c_attn_proj_b, cur), cur);
            }
        }

        if (hparams.par_res == 0) {
            struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpL);

            cur = gpt_neox_ff(model.layers[il], ctx0, inpFF, hparams.eps);

            // input for next layer
            inpL = ggml_add(ctx0, cur, inpFF);
        } else {
            struct ggml_tensor * inpFF = cur;

            // this is independent of the self-attention result, so it could be done in parallel to the self-attention
            // note here we pass inpL instead of cur
            cur = gpt_neox_ff(model.layers[il], ctx0, inpL, hparams.eps);

            // layer input + FF
            cur  = ggml_add(ctx0, cur, inpFF);

            // input for next layer
            inpL = ggml_add(ctx0, cur, inpL);
        }

    }

    // norm
    {
        inpL = ggml_norm(ctx0, inpL, hparams.eps);

        // inpL = ln_f_g*inpL + ln_f_b
        inpL = ggml_add(ctx0,
                ggml_mul(ctx0,
                    ggml_repeat(ctx0, model.ln_f_g, inpL),
                    inpL),
                ggml_repeat(ctx0, model.ln_f_b, inpL));
    }

    // lm_head
    {
        inpL = ggml_mul_mat(ctx0, model.lmh_g, inpL);

        //inpL = ggml_add(ctx0,
        //        ggml_repeat(ctx0, model.lmh_b, inpL),
        //        inpL);
    }

    // logits -> probs
    //inpL = ggml_soft_max_inplace(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result for just the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }
    //printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return true;
}

std::string execute_prompt(
        const dollyv2_model &model,
        gpt_vocab &vocab,
        const std::string &prompt,
        gpt_params &params,
        std::mt19937 &rng,
        int64_t t_load_us,
        int64_t t_sample_us,
        int64_t t_predict_us,
        size_t mem_per_token,
        int n_past,
        bool stream_response_to_cout = false) {
    std::string output = "";
    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, prompt);

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int)embd_inp.size());

    printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
    for (size_t i = 0; i < embd_inp.size(); i++) {
        printf("%s: token[%zu] = %6d, %s\n", __func__, i, embd_inp[i], vocab.id_to_token.at(embd_inp[i]).c_str());
    }
    printf("\n");

    std::vector<gpt_vocab::id> embd;

    dollyv2_eval(model, params.n_threads, 0, {0, 1, 2, 3}, logits, mem_per_token);

    const int32_t end_token = vocab.token_to_id["### End"];

    for (size_t i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!dollyv2_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) {
                printf("Failed to predict\n");
                return output;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if (i >= embd_inp.size()) {
            // sample next token
            const int top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp = params.temp;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            {
                const int64_t t_start_sample_us = ggml_time_us();

                id = gpt_sample_top_k_top_p(vocab, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);
        } else {
            // if here, it means we are still processing the input prompt
            for (size_t k = i; k < embd_inp.size(); k++) {
                embd.push_back(embd_inp[k]);
                if (int32_t(embd.size()) > params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        // display text
        for (auto id : embd) {
            output += vocab.id_to_token[id];
            if (stream_response_to_cout) {
                printf("%s", vocab.id_to_token[id].c_str());
            }
        }
        if (stream_response_to_cout) {
            fflush(stdout);
        }

        // end of text token
        if (embd.back() == 0 || (end_token > 0 && embd.back() == end_token)) {
            return output;
        }
    }
    return output;
}

#if defined(DOLLY_INTERACTIVE_PORT)
int setup_port(const int port) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        fprintf(stderr, "%s: Failed to create new socket\n", __func__);
        return -1;
    }

    sockaddr_in servaddr;
    std::memset(&servaddr, 0, sizeof(servaddr));

    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
        fprintf(stderr, "%s: Failed to bind to port %i\n", __func__, port);
        return -1;
    }

    if (listen(sockfd, 10) < 0) {
        fprintf(stderr, "%s: Failed to listen to socket on port %i\n", __func__, port);
        return -1;
    }
    return sockfd;
}

std::string read_from_port(int sockfd, int clientfd) {
    if (clientfd < 0) {
        fprintf(stderr, "%s: Failed to accept new connection\n", __func__);
        return "";
    }

    char buffer[4096];
    std::memset(buffer, 0, sizeof(buffer));

    if (read(clientfd, buffer, sizeof(buffer)) < 0) {
        fprintf(stderr, "%s: Failed to read from client\n", __func__);
    } else {
        std::cout << "Received: " << buffer;
        return std::string(buffer);
    }
    return std::string("");
}
#endif

int main(int argc, char ** argv) {
    ggml_time_init();

    const int64_t t_main_start_us = ggml_time_us();

    gpt_params params;
    params.model = "models/dolly-v2-3b/ggml-model-f16.bin";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    printf("%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);

    int64_t t_load_us = 0;
    int64_t t_sample_us = 0;
    int64_t t_predict_us = 0;

    // determine the required inference memory per token:
    size_t mem_per_token = 0;

    int n_past = 0;

    gpt_vocab vocab;
    dollyv2_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!dollyv2_model_load(params.model, model, vocab)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;

        test_gpt_tokenizer(vocab, params.token_test);
    }

#if defined(DOLLY_INTERACTIVE_PORT)
    int sockfd = -1;
    if (params.interactive_port != -1) {
        sockfd = setup_port(params.interactive_port);
        if (sockfd == -1) {
            return 1;
        }
        fprintf(stdout, "Model is ready on port %i\n", params.interactive_port);
        fflush(stdout);
    }
#endif

    if (params.interactive || params.interactive_port != -1) {
        while (true) {
            std::string prompt_input;
#if defined(DOLLY_INTERACTIVE_PORT)
            int clientfd = -1;
            if (params.interactive_port != -1) {
                sockaddr_in clientaddr;
                socklen_t clientaddrlen = sizeof(clientaddr);
                clientfd = accept(sockfd, (struct sockaddr *)&clientaddr, &clientaddrlen);
                prompt_input = read_from_port(sockfd, clientfd);
            } else
#endif
            {
                printf("Please enter your quesiton:\n>");
                fflush(stdout);

                std::getline(std::cin, prompt_input);
            }

            if (strcmp(prompt_input.c_str(), "exit") == 0) {
                break;
            }

            const std::string prompt = prompt_for_generation(prompt_input);
            // call the model
            const std::string response = execute_prompt(model, vocab, prompt, params, rng, t_load_us, t_sample_us, t_predict_us, mem_per_token, n_past, true);

#if defined(DOLLY_INTERACTIVE_PORT)
            if (params.interactive_port != -1) {
                if (write(clientfd, response.c_str(), response.size()) < 0) {
                    fprintf(stderr, "%s: Failed to write answer '%s' to client\n", __func__, response.c_str());
                }

                if (close(clientfd) < 0) {
                    fprintf(stderr, "%s: Failed to close client socket\n", __func__);
                }
            } else
#endif
            {
                printf("%s\n\n", response.c_str());
            }
            fflush(stdout);
        }
    } else {
        if (params.prompt.empty()) {
            params.prompt = gpt_random_prompt(rng);
        }

        const std::string prompt = prompt_for_generation(params.prompt);
        execute_prompt(model, vocab, prompt, params, rng, t_load_us, t_sample_us, t_predict_us, mem_per_token, n_past, true);
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us / 1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us / 1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us / 1000.0f, t_predict_us / 1000.0f / n_past);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
    }

    ggml_free(model.ctx);

#if defined(DOLLY_INTERACTIVE_PORT)
    if (params.interactive_port != -1 && close(sockfd) < 0) {
        fprintf(stderr, "%s: Failed to close server socket\n", __func__);
    }
#endif

    return 0;
}
