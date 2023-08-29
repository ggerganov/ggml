#include "ggml/ggml.h"

#include "common-ggml.h"
#include "common.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <cinttypes>
#include <map>
#include <string>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// no defaults for now
struct mpt_hparams {
    int32_t d_model      = 0;
    int32_t max_seq_len  = 0;
    int32_t n_heads      = 0;
    int32_t n_layers     = 0;
    int32_t n_vocab      = 0;
    float alibi_bias_max = 0;
    float clip_qkv       = 0;
    int32_t ftype        = 0;
    int32_t n_ctx        = 0;

};

struct mpt_layer {
    // pre normalization
    struct ggml_tensor * norm_1_weight;

    // attention
    struct ggml_tensor * c_attn_wqkv_weight;
    struct ggml_tensor * c_attn_out_proj_weight;

    // post normalization
    struct ggml_tensor * norm_2_weight;

    // ff
    struct ggml_tensor * ffn_up_proj;
    struct ggml_tensor * ffn_down_proj;
};

struct mpt_model {
    mpt_hparams hparams;

    struct ggml_tensor * wte_weight;    // position embedding
    struct ggml_tensor * norm_f_weight; // language model head

    std::vector<mpt_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct mpt_params {
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());

    int32_t seed           = -1; // RNG seed
    int32_t n_predict      = 200; // new tokens to predict
    int32_t n_batch        = 8; // batch size for prompt processing
    int32_t n_ctx          = 512;

    std::string model      = ""; // model path
    std::string prompt     = "";
    std::string token_test = "";

    bool    perplexity     = false;

    // sampling parameters
    int32_t top_k          = 0;
    float   top_p          = 1.0f;
    float   temp           = 0.8f;
    int32_t repeat_last_n  = 64;
    float   repeat_penalty = 1.02f;

};

void mpt_print_usage(int /*argc*/, char ** argv, const mpt_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: random)\n");
    fprintf(stderr, "  -f FNAME, --file FNAME\n");
    fprintf(stderr, "                        load prompt from a file\n");
    fprintf(stderr, "  -tt TOKEN_TEST, --token_test TOKEN_TEST\n");
    fprintf(stderr, "                        test tokenization\n");
    fprintf(stderr, "  -n N, --n_predict N   number of tokens to predict (default: %d)\n", params.n_predict);
    fprintf(stderr, "  --top_k N             top-k sampling (default: %d, 0 = n_vocab)\n", params.top_k);
    fprintf(stderr, "  --top_p N             top-p sampling (default: %.2f)\n", params.top_p);
    fprintf(stderr, "  --temp N              temperature (default: %.2f)\n", params.temp);
    fprintf(stderr, "  --repeat-last-n N     last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)\n", params.repeat_last_n);
    fprintf(stderr, "  --repeat-penalty N    penalize repeat sequence of tokens (default: %.2f, 1.0 = disabled)\n", (double)params.repeat_penalty);
    fprintf(stderr, "  --perplexity          compute perplexity over the prompt\n");
    fprintf(stderr, "  -c N, --ctx-size N    size of the prompt context (default: %d)\n", params.n_ctx);
    fprintf(stderr, "  -b N, --batch_size N  batch size for prompt processing (default: %d)\n", params.n_batch);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "\n");
}

bool mpt_params_parse(int argc, char ** argv, mpt_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-p" || arg == "--prompt") {
            params.prompt = argv[++i];
        } else if (arg == "-n" || arg == "--n_predict") {
            params.n_predict = std::stoi(argv[++i]);
        } else if (arg == "--top_k") {
            params.top_k = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--top_p") {
            params.top_p = std::stof(argv[++i]);
        } else if (arg == "--temp") {
            params.temp = std::stof(argv[++i]);
        } else if (arg == "--repeat-last-n") {
            params.repeat_last_n = std::stof(argv[++i]);
        } else if (arg == "--repeat-penalty") {
            params.repeat_penalty = std::stof(argv[++i]);
        } else if (arg == "--perplexity") {
            params.perplexity = true;
        } else if (arg == "-c" || arg == "--ctx-size") {
            params.n_ctx = std::stoi(argv[++i]);
        } else if (arg == "-b" || arg == "--batch_size") {
            params.n_batch = std::stoi(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            mpt_print_usage(argc, argv, params);
            exit(0);
        } else if (arg == "-f" || arg == "--file") {
            if (++i > argc) {
                fprintf(stderr, "Invalid file param");
                break;
            }
            std::ifstream file(argv[i]);
            if (!file) {
                fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
                break;
            }
            params.prompt.clear();
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
            if (params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        } else if (arg == "-tt" || arg == "--token_test") {
            params.token_test = argv[++i];
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            mpt_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

// load the model's weights from a file
bool mpt_model_load(const std::string & fname, mpt_model & model, gpt_vocab & vocab) {
    printf("%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *)&magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // load hparams
    {
        auto & hparams = model.hparams;

        fin.read((char *) &hparams.d_model,        sizeof(hparams.d_model));
        fin.read((char *) &hparams.max_seq_len,    sizeof(hparams.max_seq_len));
        fin.read((char *) &hparams.n_heads,        sizeof(hparams.n_heads));
        fin.read((char *) &hparams.n_layers,       sizeof(hparams.n_layers));
        fin.read((char *) &hparams.n_vocab,        sizeof(hparams.n_vocab));
        fin.read((char *) &hparams.alibi_bias_max, sizeof(hparams.alibi_bias_max));
        fin.read((char *) &hparams.clip_qkv,       sizeof(hparams.clip_qkv));
        fin.read((char *) &hparams.ftype,          sizeof(hparams.ftype));

        hparams.n_ctx = std::min(hparams.max_seq_len, hparams.n_ctx);

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: d_model        = %d\n", __func__, hparams.d_model);
        printf("%s: max_seq_len    = %d\n", __func__, hparams.max_seq_len);
        printf("%s: n_ctx          = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_heads        = %d\n", __func__, hparams.n_heads);
        printf("%s: n_layers       = %d\n", __func__, hparams.n_layers);
        printf("%s: n_vocab        = %d\n", __func__, hparams.n_vocab);
        printf("%s: alibi_bias_max = %f\n", __func__, hparams.alibi_bias_max);
        printf("%s: clip_qkv       = %f\n", __func__, hparams.clip_qkv);
        printf("%s: ftype          = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr          = %d\n", __func__, qntvr);

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

            // Convert token from utf-8
            std::wstring word_multibytes = convert_to_wstring(word);
            word.resize(word_multibytes.size());
            for (size_t w = 0; w < word_multibytes.size(); w++) {
                word[w] = uint8_t(word_multibytes[w]);
            }

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit
    // floats or quantized in order to save memory and also to speed up the
    // computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype)(model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n", __func__, fname.c_str(),
                model.hparams.ftype);
        return false;
    }

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    const auto & hparams = model.hparams;
    const size_t n_ctx = hparams.n_ctx;

    {
        const size_t n_embd = hparams.d_model;
        const size_t n_layer = hparams.n_layers;
        const size_t n_vocab = hparams.n_vocab;

        ctx_size += n_embd * n_vocab * ggml_type_sizef(wtype); // wte_weight
        ctx_size += n_embd * ggml_type_sizef(GGML_TYPE_F32);   // norm_f_weight

        ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32));      // ln_1_weight
        ctx_size += n_layer * (3 * n_embd * n_embd * ggml_type_sizef(wtype)); // attn_Wqkv_weight
        ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype));     // attn_out_proj_weight
        ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32));      // ln_2_weight
        ctx_size += n_layer * (4 * n_embd * n_embd * ggml_type_sizef(wtype)); // mlp_mlp_up_weight
        ctx_size += n_layer * (n_embd * n_embd * 4 * ggml_type_sizef(wtype)); // mlp_mlp_down_weight

        ctx_size += n_ctx * n_layer * n_embd * ggml_type_sizef(GGML_TYPE_F16); // memory_k
        ctx_size += n_ctx * n_layer * n_embd * ggml_type_sizef(GGML_TYPE_F16); // memory_v

        ctx_size += (1 + 6 * n_layer) * 512; // object overhead

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size / (1024.0 * 1024.0));
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

        const size_t n_embd = hparams.d_model;
        const size_t n_layer = hparams.n_layers;
        const size_t n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.wte_weight    = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);
        model.norm_f_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        // map by name
        model.tensors["transformer.wte.weight"]    = model.wte_weight;
        model.tensors["transformer.norm_f.weight"] = model.norm_f_weight;

        for (int i = 0; i < (int) n_layer; ++i) {
            auto & layer = model.layers[i];

            layer.norm_1_weight          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,     n_embd);
            layer.c_attn_wqkv_weight     = ggml_new_tensor_2d(ctx, wtype,             n_embd, 3 * n_embd);
            layer.c_attn_out_proj_weight = ggml_new_tensor_2d(ctx, wtype,             n_embd,     n_embd);
            layer.norm_2_weight          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,     n_embd);
            layer.ffn_up_proj            = ggml_new_tensor_2d(ctx, wtype,             n_embd, 4 * n_embd);
            layer.ffn_down_proj          = ggml_new_tensor_2d(ctx, wtype,         4 * n_embd,     n_embd);

            // map by name
            model.tensors["transformer.blocks." + std::to_string(i) + ".norm_1.weight"]        = layer.norm_1_weight;
            model.tensors["transformer.blocks." + std::to_string(i) + ".attn.Wqkv.weight"]     = layer.c_attn_wqkv_weight;
            model.tensors["transformer.blocks." + std::to_string(i) + ".attn.out_proj.weight"] = layer.c_attn_out_proj_weight;
            model.tensors["transformer.blocks." + std::to_string(i) + ".norm_2.weight"]        = layer.norm_2_weight;
            model.tensors["transformer.blocks." + std::to_string(i) + ".ffn.up_proj.weight"]   = layer.ffn_up_proj;
            model.tensors["transformer.blocks." + std::to_string(i) + ".ffn.down_proj.weight"] = layer.ffn_down_proj;
        }
    }

    // key + value memory
    {
        const auto & hparams = model.hparams;

        const size_t n_embd  = hparams.d_model;
        const size_t n_layer = hparams.n_layers;

        const int64_t n_mem      = n_layer * n_ctx;
        const int64_t n_elements = n_embd  * n_mem;

        model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);
        model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);

        const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

        printf("%s: memory_size = %8.2f MB, n_mem = %" PRId64 "\n", __func__, memory_size / 1024.0 / 1024.0, n_mem);
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
            fin.read(reinterpret_cast<char *>(&ttype), sizeof(ttype));

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = {1, 1};
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
                fprintf(stderr,
                        "%s: tensor '%s' has wrong shape in model file: got [%5d, "
                        "%5d], expected [%5d, %5d]\n",
                        __func__, name.c_str(), (int)tensor->ne[0], (int)tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            // for debugging
            if (0) {
                printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", name.c_str(), ne[0], ne[1],
                       ggml_type_name(ggml_type(ttype)), ggml_nbytes(tensor) / 1024.0 / 1024.0, ggml_nbytes(tensor));
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr,
                        "%s: tensor '%s' has wrong size in model file: got %zu, "
                        "expected %zu\n",
                        __func__, name.c_str(), ggml_nbytes(tensor), nelements * bpe);
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

        printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0, n_tensors);
    }

    fin.close();

    return true;
}

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
bool mpt_eval(const mpt_model & model, const int n_threads, const int n_past,
              const std::vector<gpt_vocab::id> & embd_inp, std::vector<float> & embd_w, bool logits_all, size_t & mem_per_token) {
    const int N = embd_inp.size();

    const auto & hparams = model.hparams;

    const int n_embd  = hparams.d_model;
    const int n_layer = hparams.n_layers;
    const int n_head  = hparams.n_heads;
    const int n_vocab = hparams.n_vocab;
    const int n_ctx   = hparams.n_ctx;
    const float eps   = 1e-5f;

    static size_t buf_size = 256u * 1024 * 1024;
    static void * buf = malloc(buf_size);

    // use 2 scratch buffers
    // TODO: very hacky solution - reimplement in a more elegant way
    static size_t scr0_size = 256u*1024*1024;
    static void * scr0 = malloc(scr0_size);

    static size_t scr1_size = 256u*1024*1024;
    static void * scr1 = malloc(scr1_size);

    if (mem_per_token > 0 && mem_per_token * N > buf_size) {
        const size_t buf_size_new = 1.1 * (mem_per_token * N); // add 10% to account for ggml object overhead
        // printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__,
        // buf_size, buf_size_new);

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
    struct ggml_cgraph gf = {};

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, embd_inp.data(), N * ggml_element_size(embd));

    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.wte_weight, embd);

    for (int il = 0; il < n_layer; ++il) {

        struct ggml_tensor * cur;

        ggml_set_scratch(ctx0, { 0, scr0_size, scr0, });

        // a = self.ln_1(x)
        {
            cur = ggml_norm(ctx0, inpL, eps);

            cur = ggml_mul(ctx0, ggml_repeat(ctx0, model.layers[il].norm_1_weight, cur), cur);
        }

        // self-attention
        //  b, _, past_key_value = self.attn(a, past_key_value=past_key_value,
        //  attn_bias=attn_bias, attention_mask=attention_mask,
        //  is_causal=is_causal)
        {
            // compute QKV
            cur = ggml_mul_mat(ctx0, model.layers[il].c_attn_wqkv_weight, cur);

            if (model.hparams.clip_qkv > 0.0f) {
                cur = ggml_clamp(ctx0, cur, -model.hparams.clip_qkv, model.hparams.clip_qkv);
            }

            struct ggml_tensor * Qcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 0 * sizeof(float) * n_embd);
            struct ggml_tensor * Kcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 1 * sizeof(float) * n_embd);
            struct ggml_tensor * Vcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 2 * sizeof(float) * n_embd);

            // store key and value to memory
            {
                struct ggml_tensor * k =
                    ggml_view_1d(ctx0, model.memory_k, N * n_embd,
                                 (ggml_element_size(model.memory_k) * n_embd) * (il * n_ctx + n_past));
                struct ggml_tensor * v =
                    ggml_view_1d(ctx0, model.memory_v, N * n_embd,
                                 (ggml_element_size(model.memory_v) * n_embd) * (il * n_ctx + n_past));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0,
            // 2, 1, 3) [64, N, 12]
            struct ggml_tensor * Q = ggml_permute(
                ctx0, ggml_cpy(ctx0, Qcur, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd / n_head, n_head, N)), 0, 2,
                1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1,
            // 3) [64, n_past + N, 12]
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                             ggml_reshape_3d(ctx0,
                                             ggml_view_1d(ctx0, model.memory_k, (n_past + N) * n_embd,
                                                          il * n_ctx * ggml_element_size(model.memory_k) * n_embd),
                                             n_embd / n_head, n_head, n_past + N),
                             0, 2, 1, 3);
            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor * KQ_scaled =
                ggml_scale(ctx0, KQ, ggml_new_f32(ctx0, 1.0f / sqrt(float(n_embd) / n_head)));

            struct ggml_tensor * KQ_scaled_alibi =
                ggml_alibi(ctx0, KQ_scaled, n_past, n_head, model.hparams.alibi_bias_max);

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled_alibi, n_past);

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1,
            // 2, 0, 3).contiguous() [n_past + N, 64, 12]
            struct ggml_tensor * V_trans = ggml_cpy(
                ctx0,
                ggml_permute(ctx0,
                             ggml_reshape_3d(ctx0,
                                             ggml_view_1d(ctx0, model.memory_v, (n_past + N) * n_embd,
                                                          il * n_ctx * ggml_element_size(model.memory_v) * n_embd),
                                             n_embd / n_head, n_head, n_past + N),
                             1, 2, 0, 3),
                ggml_new_tensor_3d(ctx0, model.memory_v->type, n_past + N, n_embd / n_head, n_head));

            // KQV = transpose(V) * KQ_soft_max
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0, KQV_merged, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection
            { cur = ggml_mul_mat(ctx0, model.layers[il].c_attn_out_proj_weight, cur); }
        }

        inpL = ggml_add(ctx0, inpL, cur);

        ggml_set_scratch(ctx0, { 0, scr1_size, scr1, });

        // m = self.ln_2(x)
        {
            cur = ggml_norm(ctx0, inpL, eps);

            cur = ggml_mul(ctx0, ggml_repeat(ctx0, model.layers[il].norm_2_weight, cur), cur);
        }

        // n = self.mlp(m)
        {

            cur = ggml_mul_mat(ctx0, model.layers[il].ffn_up_proj, cur);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // projection
            // cur = proj_w*cur + proj_b
            cur = ggml_mul_mat(ctx0, model.layers[il].ffn_down_proj, cur);
        }

        // x = x + n
        inpL = ggml_add(ctx0, inpL, cur);
    }

    ggml_set_scratch(ctx0, { 0, scr0_size, scr0, });

    // norm
    {
        inpL = ggml_norm(ctx0, inpL, eps);
        // inpL = ln_f_g*inpL
        inpL = ggml_mul(ctx0, ggml_repeat(ctx0, model.norm_f_weight, inpL), inpL);
    }

    ggml_set_scratch(ctx0, { 0, 0, nullptr, });

    // output embedding weight tied to input embedding
    inpL = ggml_mul_mat(ctx0, model.wte_weight, inpL);

    // logits -> probs
    // inpL = ggml_soft_max(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

    // std::cout << "Qcur" << std::endl;
    // print_tensor(Qcur);

    // if (n_past%100 == 0) {
    // ggml_graph_print(&gf);
    // ggml_graph_dump_dot(&gf, NULL, "mpt-model.dot");
    // }

    if (logits_all) {
        // return result for all tokens
        embd_w.resize(n_vocab *N);
        memcpy(embd_w.data(), (float *)ggml_get_data(inpL) , sizeof(float) * n_vocab * N);
    } else {
        // return result for just the last token
        embd_w.resize(n_vocab);
        memcpy(embd_w.data(), (float *)ggml_get_data(inpL) + (n_vocab * (N - 1)), sizeof(float) * n_vocab);
    }

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0) / N;
    }
    // printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return true;
}

std::vector<float> softmax(const std::vector<float> & logits) {
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

int perplexity(const mpt_params & params) {
    ggml_time_init();

    const int64_t t_main_start_us = ggml_time_us();

    printf("%s: n_threads = %d\n", __func__, params.n_threads);
    printf("%s: n_batch   = %d\n", __func__, params.n_batch);
    printf("%s: n_ctx     = %d\n", __func__, params.n_ctx);
    printf("\n");

    int64_t t_load_us = 0;

    gpt_vocab vocab;
    mpt_model model;

    model.hparams.n_ctx = params.n_ctx;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!mpt_model_load(params.model, model, vocab)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<int> embd_inp = ::gpt_tokenize(vocab, params.prompt);

    printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    mpt_eval(model, params.n_threads, 0, {0, 1, 2, 3}, logits, false, mem_per_token);

    int count   = 0;

    const int n_chunk = embd_inp.size() / params.n_ctx;

    const int n_vocab = model.hparams.n_vocab;
    const int n_batch = params.n_batch;

    double nll = 0.0;
    fprintf(stderr, "%s: calculating perplexity over %d chunks, batch_size=%d\n", __func__, n_chunk, n_batch);

    for (int i = 0; i < n_chunk; ++i) {

        const int start =     i * params.n_ctx;
        const int end   = start + params.n_ctx;

        const int num_batches = (params.n_ctx + n_batch - 1) / n_batch;

        std::vector<float> logits;

        const auto t_start = std::chrono::high_resolution_clock::now();

        for (int j = 0; j < num_batches; ++j) {

            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            std::vector<gpt_vocab::id> embd;

            for(int p=0;p<batch_size;p++) {
                embd.push_back( embd_inp[batch_start+p]  );
            }

            std::vector<float> batch_logits;// = llama_get_logits(ctx);

            const int64_t t_start_us = ggml_time_us();

            if (!mpt_eval(model, params.n_threads, j * batch_size, embd, batch_logits, true, mem_per_token)) {
                printf("%s: failed to evaluate model\n", __func__);
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;

            logits.insert(logits.end(), batch_logits.data(), batch_logits.data() + batch_size * n_vocab);

        }

        const auto t_end = std::chrono::high_resolution_clock::now();

        if (i == 0) {
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            fprintf(stderr, "%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total * n_chunk);
            if (total_seconds >= 60*60) {
                fprintf(stderr, "%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            fprintf(stderr, "%d minutes\n", total_seconds / 60);

            printf("\nChunk\tPPL cumulative\tPPL chunk\n");
        }

        // We get the logits for all the tokens in the context window (params.n_ctx)
        // from llama_eval above.  Now, based on https://huggingface.co/docs/transformers/perplexity,
        // calculate the perplexity over the last half of the window (so the model always has
        // some context to predict the token).
        //
        // We rely on the fact that attention in the forward pass only looks at previous
        // tokens here, so the logits returned for each token are an accurate representation
        // of what the model would have predicted at that point.
        //
        // Example, we have a context window of 512, we will compute perplexity for each of the
        // last 256 tokens.  Then, we split the input up into context window size chunks to
        // process the entire prompt.

        double nllchunk = 0.0;
        int countchunk = 0;

        for (int j = std::min(512, params.n_ctx / 2); j < params.n_ctx - 1; ++j) {
            // Calculate probability of next token, given the previous ones.
            const std::vector<float> tok_logits(
                logits.begin() + (j + 0) * n_vocab,
                logits.begin() + (j + 1) * n_vocab);

            const float prob = softmax(tok_logits)[embd_inp[ start+ j + 1]];

            nllchunk += -std::log(prob);
            ++countchunk;
        }

		nll += nllchunk;
		count += countchunk;

        // perplexity is e^(average negative log-likelihood)
        printf("%d\t%.8lf\t%.8lf\n", i + 1, std::exp(nll / count), std::exp(nllchunk/countchunk) );
        fflush(stdout);
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n",   __func__, t_load_us / 1000.0f);
        printf("%s:     eval time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us / 1000.0f, t_predict_us / 1000.0f / (n_chunk * params.n_ctx));
        printf("%s:    total time = %8.2f ms\n",   __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}

int main(int argc, char ** argv) {
    mpt_params params;

    if (mpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.perplexity) {
        return perplexity(params);
    }

    ggml_time_init();

    const int64_t t_main_start_us = ggml_time_us();

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    if (params.n_predict < 0) {
        params.n_predict = 0;
    }

    printf("%s: seed      = %d\n",   __func__, params.seed);
    printf("%s: n_threads = %d\n",   __func__, params.n_threads);
    printf("%s: n_batch   = %d\n",   __func__, params.n_batch);
    printf("%s: n_ctx     = %d\n",   __func__, params.n_ctx);
    printf("%s: n_predict = %d\n\n", __func__, params.n_predict);

    std::mt19937 rng(params.seed);
    if (params.prompt.empty()) {
        params.prompt = gpt_random_prompt(rng);
    }

    int64_t t_load_us = 0;

    gpt_vocab vocab;
    mpt_model model;

    model.hparams.n_ctx = params.n_ctx;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!mpt_model_load(params.model, model, vocab)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;

        test_gpt_tokenizer(vocab, params.token_test);
    }

    if (params.top_k == 0) {
        params.top_k = model.hparams.n_vocab;
    }

    if (params.repeat_last_n == -1) {
        params.repeat_last_n = params.n_ctx;
    }

    printf("\n");
    printf("%s: temp           = %.3f\n", __func__, params.temp);
    printf("%s: top_k          = %d\n",   __func__, params.top_k);
    printf("%s: top_p          = %.3f\n", __func__, params.top_p);
    printf("%s: repeat_last_n  = %d\n",   __func__, params.repeat_last_n);
    printf("%s: repeat_penalty = %.3f\n", __func__, params.repeat_penalty);

    int64_t t_sample_us = 0;
    int64_t t_predict_us = 0;

    std::vector<int32_t> last_n_tokens(params.n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    // tokenize the prompt
    std::vector<int> embd_inp = ::gpt_tokenize(vocab, params.prompt);

    printf("\n");
    printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());

    for (size_t i = 0; i < embd_inp.size(); i++) {
        printf("%s: token[%zu] = %6d\n", __func__, i, embd_inp[i]);
    }
    printf("\n");

    std::vector<gpt_vocab::id> embd;
    std::vector<float> logits;

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    mpt_eval(model, params.n_threads, 0, {0, 1, 2, 3}, logits, false, mem_per_token);

    int n_past     = 0;
    int n_consumed = 0;
    int n_sampled  = 0;

    while (n_sampled < params.n_predict) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!mpt_eval(model, params.n_threads, n_past, embd, logits, false, mem_per_token)) {
                printf("%s: failed to predict\n", __func__);
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;

            n_past += embd.size();
            embd.clear();
        }

        if ((int)embd_inp.size() <= n_consumed) {
            // sample next token

            const int top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp = params.temp;
            const int repeat_last_n = params.repeat_last_n;
            const float repeat_penalty = params.repeat_penalty;

            gpt_vocab::id id = 0;

            {
                const int64_t t_start_sample_us = ggml_time_us();

                id = gpt_sample_top_k_top_p_repeat(vocab, logits.data() + (logits.size() - model.hparams.n_vocab), last_n_tokens.data(), last_n_tokens.size(), top_k, top_p, temp, repeat_last_n, repeat_penalty, rng);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);
            ++n_sampled;

        } else {
            // if here, it means we are still processing the input prompt
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        for (auto id : embd) {
           printf("%s", vocab.id_to_token[id].c_str());
        }
        fflush(stdout);

        // end of text token
        if (embd.back() == 0) {
            break;
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n\n");
        printf("%s: sampled tokens = %8d\n", __func__, n_sampled);
        printf("%s:  mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:      load time = %8.2f ms\n", __func__, t_load_us / 1000.0f);
        printf("%s:    sample time = %8.2f ms / %.2f ms per token\n", __func__, t_sample_us / 1000.0f, t_sample_us / 1000.0f / n_sampled);
        printf("%s:      eval time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us / 1000.0f, t_predict_us / 1000.0f / n_past);
        printf("%s:     total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}
