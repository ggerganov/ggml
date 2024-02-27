#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "common.h"
#include "common-ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define GPT2_MAX_NODES 4096

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

typedef int32_t gpt2_pos;
typedef int32_t gpt2_seq_id;

// default hparams (GPT-2 117M)
struct gpt2_hparams {
    int32_t n_vocab = 50257;
    int32_t n_ctx   = 1024;
    int32_t n_embd  = 768;
    int32_t n_head  = 12;
    int32_t n_layer = 12;
    int32_t ftype   = 1;
    float   eps     = 1e-5f;
};

struct gpt2_layer {
    // normalization
    struct ggml_tensor * ln_1_g;
    struct ggml_tensor * ln_1_b;

    struct ggml_tensor * ln_2_g;
    struct ggml_tensor * ln_2_b;

    // attention
    struct ggml_tensor * c_attn_attn_w;
    struct ggml_tensor * c_attn_attn_b;

    struct ggml_tensor * c_attn_proj_w;
    struct ggml_tensor * c_attn_proj_b;

    // mlp
    struct ggml_tensor * c_mlp_fc_w;
    struct ggml_tensor * c_mlp_fc_b;

    struct ggml_tensor * c_mlp_proj_w;
    struct ggml_tensor * c_mlp_proj_b;
};

struct gpt2_kv_cell {
    gpt2_pos pos   = -1;
    gpt2_pos delta = 0;

    std::set<gpt2_seq_id> seq_id;

    bool has_seq_id(const gpt2_seq_id & id) const {
        return seq_id.find(id) != seq_id.end();
    }
};

struct gpt2_kv_cache {
    // key + value memory
    struct ggml_tensor * k;
    struct ggml_tensor * v;
    //

    uint32_t head = 0;
    uint32_t size = 0;

    // computed before each graph build
    uint32_t n = 0;

    std::vector<gpt2_kv_cell> cells;

    ggml_backend_buffer_t buffer;
};

struct gpt2_model {
    gpt2_hparams hparams;

    // normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * wte;     // position embedding
    struct ggml_tensor * wpe;     //    token embedding
    struct ggml_tensor * lm_head; // language model head

    std::vector<gpt2_layer> layers;

    gpt2_kv_cache kv_cache;

    struct ggml_context * ctx_w;

    ggml_backend_t backend = NULL;

    ggml_backend_buffer_t buffer_w;

    std::map<std::string, struct ggml_tensor *> tensors;
};

// Input data for gpt2_decode
// A gpt2_batch object can contain input about one or many sequences
// The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
//
// - token  : the token ids of the input (used when embd is NULL)
// - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
// - pos    : the positions of the respective token in the sequence
// - seq_id : the sequence to which the respective token belongs
// - logits : if zero, the logits for the respective token will not be output
//
struct gpt2_batch {
    int32_t n_tokens = -1;

    gpt_vocab::id  * token  = {};
    float          * embd   = {};
    gpt2_pos       * pos    = {};
    gpt2_seq_id    * seq_id = {};
    int8_t         * logits = {};
};

// load the model's weights from a file
bool gpt2_model_load(const std::string & fname, gpt2_model & model, gpt_vocab & vocab, int n_ctx, int n_gpu_layers) {
    printf("%s: loading model from '%s'\n", __func__, fname.c_str());

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
        fin.read((char *) &hparams.ftype,   sizeof(hparams.ftype));

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: ftype   = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr   = %d\n", __func__, qntvr);

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        fin.read((char *) &n_vocab, sizeof(n_vocab));

        if (n_vocab != model.hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
            return false;
        }

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
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype) (model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
                __func__, fname.c_str(), model.hparams.ftype);
        return false;
    }

    auto & ctx = model.ctx_w;

    size_t buffer_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        buffer_size += ggml_row_size(GGML_TYPE_F32, n_embd); // ln_f_g
        buffer_size += ggml_row_size(GGML_TYPE_F32, n_embd); // ln_f_b

        buffer_size += ggml_row_size(wtype,         n_vocab*n_embd); // wte
        buffer_size += ggml_row_size(GGML_TYPE_F32,   n_ctx*n_embd); // wpe
        buffer_size += ggml_row_size(wtype,         n_vocab*n_embd); // lm_head

        buffer_size += n_layer*(ggml_row_size(GGML_TYPE_F32, n_embd)); // ln_1_g
        buffer_size += n_layer*(ggml_row_size(GGML_TYPE_F32, n_embd)); // ln_1_b

        buffer_size += n_layer*(ggml_row_size(GGML_TYPE_F32, n_embd)); // ln_2_g
        buffer_size += n_layer*(ggml_row_size(GGML_TYPE_F32, n_embd)); // ln_2_b

        buffer_size += n_layer*(ggml_row_size(wtype,         3*n_embd*n_embd)); // c_attn_attn_w
        buffer_size += n_layer*(ggml_row_size(GGML_TYPE_F32, 3*n_embd));        // c_attn_attn_b

        buffer_size += n_layer*(ggml_row_size(wtype,         n_embd*n_embd));   // c_attn_proj_w
        buffer_size += n_layer*(ggml_row_size(GGML_TYPE_F32, n_embd));          // c_attn_proj_b

        buffer_size += n_layer*(ggml_row_size(wtype,         4*n_embd*n_embd)); // c_mlp_fc_w
        buffer_size += n_layer*(ggml_row_size(GGML_TYPE_F32, 4*n_embd));        // c_mlp_fc_b

        buffer_size += n_layer*(ggml_row_size(wtype,         4*n_embd*n_embd)); // c_mlp_proj_w
        buffer_size += n_layer*(ggml_row_size(GGML_TYPE_F32, 4*n_embd));        // c_mlp_proj_b

        buffer_size += (6 + 12*n_layer)*128; // alignment overhead

        printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
        printf("%s: backend buffer size = %6.2f MB\n", __func__, buffer_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        size_t n_tensors = 2 + 6 + 12*model.hparams.n_layer;
        struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };

        model.ctx_w = ggml_init(params);
        if (!model.ctx_w) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // initialize the backend
#ifdef GGML_USE_CUBLAS
    if (n_gpu_layers > 0) {
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init(0);
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (n_gpu_layers > 0) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        ggml_backend_metal_log_set_callback(ggml_log_callback_default, nullptr);
        model.backend = ggml_backend_metal_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#endif

    if (!model.backend) {
        // fallback to CPU backend
        fprintf(stderr, "%s: using CPU backend\n", __func__);
        model.backend = ggml_backend_cpu_init();
    }

    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_cpu_init() failed\n", __func__);
        return false;
    }

    // allocate weights buffer
    model.buffer_w = ggml_backend_alloc_buffer(model.backend, buffer_size);

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.ln_f_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.ln_f_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        model.wte     = ggml_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);
        model.wpe     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ctx);
        model.lm_head = ggml_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);

        // map by name
        model.tensors["model/ln_f/g"] = model.ln_f_g;
        model.tensors["model/ln_f/b"] = model.ln_f_b;

        model.tensors["model/wte"]     = model.wte;
        model.tensors["model/wpe"]     = model.wpe;
        model.tensors["model/lm_head"] = model.lm_head;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            layer.ln_1_g        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.ln_1_b        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.ln_2_g        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.ln_2_b        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.c_attn_attn_w = ggml_new_tensor_2d(ctx, wtype,           n_embd, 3*n_embd);
            layer.c_attn_attn_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3*n_embd);

            layer.c_attn_proj_w = ggml_new_tensor_2d(ctx, wtype,           n_embd, n_embd);
            layer.c_attn_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.c_mlp_fc_w    = ggml_new_tensor_2d(ctx, wtype,           n_embd, 4*n_embd);
            layer.c_mlp_fc_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*n_embd);

            layer.c_mlp_proj_w  = ggml_new_tensor_2d(ctx, wtype,         4*n_embd, n_embd);
            layer.c_mlp_proj_b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            // map by name
            model.tensors["model/h" + std::to_string(i) + "/ln_1/g"]        = layer.ln_1_g;
            model.tensors["model/h" + std::to_string(i) + "/ln_1/b"]        = layer.ln_1_b;

            model.tensors["model/h" + std::to_string(i) + "/ln_2/g"]        = layer.ln_2_g;
            model.tensors["model/h" + std::to_string(i) + "/ln_2/b"]        = layer.ln_2_b;

            model.tensors["model/h" + std::to_string(i) + "/attn/c_attn/w"] = layer.c_attn_attn_w;
            model.tensors["model/h" + std::to_string(i) + "/attn/c_attn/b"] = layer.c_attn_attn_b;

            model.tensors["model/h" + std::to_string(i) + "/attn/c_proj/w"] = layer.c_attn_proj_w;
            model.tensors["model/h" + std::to_string(i) + "/attn/c_proj/b"] = layer.c_attn_proj_b;

            model.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/w"]    = layer.c_mlp_fc_w;
            model.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/b"]    = layer.c_mlp_fc_b;

            model.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/w"]  = layer.c_mlp_proj_w;
            model.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/b"]  = layer.c_mlp_proj_b;
        }
    }

    // override the default training context with the user-provided
    model.hparams.n_ctx = n_ctx;

    // key + value memory
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;

        const int n_mem      = n_layer*n_ctx;
        const int n_elements = n_embd*n_mem;

        model.kv_cache.k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        model.kv_cache.v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);

        model.kv_cache.head      = 0;
        model.kv_cache.size      = n_ctx;

        model.kv_cache.cells.resize(n_ctx);

        const size_t memory_size = ggml_nbytes(model.kv_cache.k) + ggml_nbytes(model.kv_cache.v);

        printf("%s: memory size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);

        // create a backend buffer (can be in host or device memory)
        model.kv_cache.buffer = ggml_backend_alloc_buffer(model.backend, memory_size + 256);

        // allocate the tensors into the backend buffer
        {
            ggml_tallocr * alloc = ggml_tallocr_new(model.kv_cache.buffer);

            // this updates the pointers in the tensors to point to the correct location in the buffer
            // this is necessary since the ggml_context is .no_alloc == true
            // note that the buffer can actually be a device buffer, depending on the backend
            ggml_tallocr_alloc(alloc, model.kv_cache.k);
            ggml_tallocr_alloc(alloc, model.kv_cache.v);

            ggml_tallocr_free(alloc);
        }
    }

    // load weights
    {
        ggml_tallocr * alloc = ggml_tallocr_new(model.buffer_w);

        size_t total_size = 0;

        bool has_lm_head = false;

        std::vector<char> read_buf;

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
            ggml_set_name(tensor, name.c_str());
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.c_str());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
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

            ggml_tallocr_alloc(alloc, tensor);

            if (ggml_backend_is_cpu  (model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
                ) {
                // for the CPU and Metal backend, we can read directly into the tensor
                fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(ggml_nbytes(tensor));
                fin.read(read_buf.data(), ggml_nbytes(tensor));
                ggml_backend_tensor_set(tensor, read_buf.data(), 0, ggml_nbytes(tensor));
            }

            // GPT-2 models share the WTE tensor as the LM head
            if (name == "model/wte" && has_lm_head == false) {
                //ggml_tallocr_alloc(alloc, model.lm_head);
                //ggml_backend_tensor_copy(tensor, model.lm_head);
                model.lm_head = tensor;
            }

            if (name == "model/lm_head") {
                has_lm_head = true;
            }

            total_size += ggml_nbytes(tensor);
        }

        ggml_tallocr_free(alloc);
        printf("%s: model size  = %8.2f MB\n", __func__, total_size/1024.0/1024.0);
    }

    fin.close();

    return true;
}

// build the computation graph
struct ggml_cgraph * gpt2_graph(
        const  gpt2_model  & model,
        const  gpt2_batch  & batch,
                     bool    measure) {
    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;

    const auto & kv_cache = model.kv_cache;

    const int32_t n_tokens = batch.n_tokens;
    const int32_t n_kv     = measure ? n_ctx            : kv_cache.n;
    const int32_t kv_head  = measure ? n_ctx - n_tokens : kv_cache.head;

    // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
    static size_t buf_size = ggml_tensor_overhead()*GPT2_MAX_NODES + ggml_graph_overhead_custom(GPT2_MAX_NODES, false);
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };

    struct ggml_context * ctx = ggml_init(params);

    struct ggml_cgraph  * gf = ggml_new_graph_custom(ctx, GPT2_MAX_NODES, false);

    struct ggml_tensor * inpL;
    if (batch.token) {
        struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_name(inp_tokens, "inp_tokens");
        ggml_set_input(inp_tokens);

        struct ggml_tensor * position = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_name(position, "position");
        ggml_set_input(position);

        // wte + wpe
        inpL =
            ggml_add(ctx,
                    ggml_get_rows(ctx, model.wte, inp_tokens),
                    ggml_get_rows(ctx, model.wpe, position));
    } else {
        GGML_ASSERT(batch.embd);

        inpL = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_name(inpL, "embd");
        ggml_set_input(inpL);
    }

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_kv, n_tokens, 1);
    ggml_set_name(KQ_mask, "KQ_mask");
    ggml_set_input(KQ_mask);


    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur;

        // norm
        {
            // [ 768, N]
            cur = ggml_norm(ctx, inpL, hparams.eps);

            // cur = ln_1_g*cur + ln_1_b
            // [ 768, N]
            cur = ggml_add(ctx,
                    ggml_mul(ctx,
                        cur,
                        model.layers[il].ln_1_g),
                    model.layers[il].ln_1_b);
        }

        // attn
        // [2304,        768] - model.layers[il].c_attn_attn_w
        // [2304,          1] - model.layers[il].c_attn_attn_b
        // [ 768,   n_tokens] - cur (in)
        // [2304,   n_tokens] - cur (out)
        //
        // cur = attn_w*cur + attn_b
        // [2304, n_tokens]
        {
            cur = ggml_mul_mat(ctx,
                    model.layers[il].c_attn_attn_w,
                    cur);

            cur = ggml_add(ctx,
                    cur,
                    model.layers[il].c_attn_attn_b);
        }

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_view_2d(ctx, cur, n_embd, n_tokens, cur->nb[1], 0*sizeof(float)*n_embd);
            struct ggml_tensor * Kcur = ggml_view_2d(ctx, cur, n_embd, n_tokens, cur->nb[1], 1*sizeof(float)*n_embd);
            struct ggml_tensor * Vcur = ggml_view_2d(ctx, cur, n_embd, n_tokens, cur->nb[1], 2*sizeof(float)*n_embd);

            // store key and value to memory
            if (n_tokens >= 1) {
                struct ggml_tensor * k = ggml_view_1d(ctx, model.kv_cache.k, n_tokens*n_embd, (ggml_element_size(model.kv_cache.k)*n_embd)*(il*n_ctx + kv_head));
                struct ggml_tensor * v = ggml_view_1d(ctx, model.kv_cache.v, n_tokens*n_embd, (ggml_element_size(model.kv_cache.v)*n_embd)*(il*n_ctx + kv_head));

                ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            // [64, N, 12]
            struct ggml_tensor * Q =
                ggml_permute(ctx,
                        ggml_cont_3d(ctx,
                            Qcur,
                            n_embd/n_head, n_head, n_tokens),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_kv).permute(0, 2, 1, 3)
            // [64, n_kv, 12]
            struct ggml_tensor * K =
                ggml_permute(ctx,
                        ggml_reshape_3d(ctx,
                            ggml_view_1d(ctx, model.kv_cache.k, n_kv*n_embd, il*n_ctx*ggml_element_size(model.kv_cache.k)*n_embd),
                            n_embd/n_head, n_head, n_kv),
                        0, 2, 1, 3);

            // GG: flash attention
            //struct ggml_tensor * V =
            //    ggml_cpy(ctx0,
            //            ggml_permute(ctx0,
            //                ggml_reshape_3d(ctx0,
            //                    ggml_view_1d(ctx0, model.kv_cache.v, n_kv*n_embd, il*n_ctx*ggml_element_size(model.kv_cache.v)*n_embd),
            //                    n_embd/n_head, n_head, n_kv),
            //                1, 2, 0, 3),
            //            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_kv, n_embd/n_head, n_head));

            //struct ggml_tensor * KQV = ggml_flash_attn(ctx0, Q, K, V, true);

            // K * Q
            // [n_kv, n_tokens, 12]
            struct ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // [n_kv, n_tokens, 12]
            struct ggml_tensor * KQ_scaled =
                ggml_scale(ctx,
                        KQ,
                        1.0f/sqrtf(float(n_embd)/n_head));

            // KQ_masked = mask_past(KQ_scaled)
            // [n_kv, n_tokens, 12]
            struct ggml_tensor * KQ_masked = ggml_add(ctx, KQ_scaled, KQ_mask);

            // KQ = soft_max(KQ_masked)
            // [n_kv, N, 12]
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_kv).permute(1, 2, 0, 3).contiguous()
            // [n_kv, 64, 12]
            struct ggml_tensor * V_trans =
                ggml_cont_3d(ctx,
                        ggml_permute(ctx,
                            ggml_reshape_3d(ctx,
                                ggml_view_1d(ctx, model.kv_cache.v, n_kv*n_embd, il*n_ctx*ggml_element_size(model.kv_cache.v)*n_embd),
                                n_embd/n_head, n_head, n_kv),
                            1, 2, 0, 3),
                        n_kv, n_embd/n_head, n_head);

            // KQV = transpose(V) * KQ_soft_max
            // [64, n_tokens, 12]
            struct ggml_tensor * KQV = ggml_mul_mat(ctx, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // [64, 12, n_tokens]
            struct ggml_tensor * KQV_merged = ggml_permute(ctx, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // [768, n_tokens]
            cur = ggml_cont_2d(ctx, KQV_merged, n_embd, n_tokens);
        }

        // projection
        // [ 768, 768] - model.layers[il].c_attn_proj_w
        // [ 768,   1] - model.layers[il].c_attn_proj_b
        // [ 768,   N] - cur (in)
        // [ 768,   N] - cur (out)
        //
        // cur = proj_w*cur + proj_b
        // [768, N]
        {
            cur = ggml_mul_mat(ctx,
                    model.layers[il].c_attn_proj_w,
                    cur);

            cur = ggml_add(ctx,
                    cur,
                    model.layers[il].c_attn_proj_b);
        }

        // add the input
        cur = ggml_add(ctx, cur, inpL);

        struct ggml_tensor * inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx, inpFF, hparams.eps);

                // cur = ln_2_g*cur + ln_2_b
                // [ 768, N]
                cur = ggml_add(ctx,
                        ggml_mul(ctx,
                            cur,
                            model.layers[il].ln_2_g),
                        model.layers[il].ln_2_b);
            }

            // fully connected
            // [3072, 768] - model.layers[il].c_mlp_fc_w
            // [3072,   1] - model.layers[il].c_mlp_fc_b
            // [ 768,   N] - cur (in)
            // [3072,   N] - cur (out)
            //
            // cur = fc_w*cur + fc_b
            // [3072, N]
            cur = ggml_mul_mat(ctx,
                    model.layers[il].c_mlp_fc_w,
                    cur);

            cur = ggml_add(ctx,
                    cur,
                    model.layers[il].c_mlp_fc_b);

            // GELU activation
            // [3072, N]
            cur = ggml_gelu(ctx, cur);

            // projection
            // [ 768, 3072] - model.layers[il].c_mlp_proj_w
            // [ 768,    1] - model.layers[il].c_mlp_proj_b
            // [3072,    N] - cur (in)
            // [ 768,    N] - cur (out)
            //
            // cur = proj_w*cur + proj_b
            // [768, N]
            cur = ggml_mul_mat(ctx,
                    model.layers[il].c_mlp_proj_w,
                    cur);

            cur = ggml_add(ctx,
                    cur,
                    model.layers[il].c_mlp_proj_b);
        }

        // input for next layer
        inpL = ggml_add(ctx, cur, inpFF);
    }

    // norm
    {
        // [ 768, N]
        inpL = ggml_norm(ctx, inpL, hparams.eps);

        // inpL = ln_f_g*inpL + ln_f_b
        // [ 768, N]
        inpL = ggml_add(ctx,
                ggml_mul(ctx,
                    inpL,
                    model.ln_f_g),
                model.ln_f_b);
    }

    // inpL = WTE * inpL
    // [ 768, 50257] - model.lm_head
    // [ 768, N]     - inpL
    inpL = ggml_mul_mat(ctx, model.lm_head, inpL);

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

    ggml_build_forward_expand(gf, inpL);

    ggml_free(ctx);

    return gf;
}

static void gpt2_kv_cache_seq_cp(
        struct gpt2_kv_cache & cache,
                 gpt2_seq_id   seq_id_src,
                 gpt2_seq_id   seq_id_dst,
                    gpt2_pos   p0,
                    gpt2_pos   p1) {
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<gpt2_pos>::max();

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id_src) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.cells[i].seq_id.insert(seq_id_dst);
        }
    }
}

struct gpt2_batch gpt2_batch_init(int32_t n_tokens, int32_t embd) {
    gpt2_batch batch;

    if (embd) {
        batch.embd = (float *) malloc(sizeof(float) * n_tokens * embd);
    } else {
        batch.token = (gpt_vocab::id *) malloc(sizeof(gpt_vocab::id) * n_tokens);
    }

    batch.pos    = (gpt2_pos *)    malloc(sizeof(gpt2_pos)    * n_tokens);
    batch.seq_id = (gpt2_seq_id *) malloc(sizeof(gpt2_seq_id) * n_tokens);
    batch.logits = (int8_t *)      malloc(sizeof(int8_t)      * n_tokens);

    return batch;
}

void gpt2_batch_free(struct gpt2_batch batch) {
    if (batch.token)  free(batch.token);
    if (batch.embd)   free(batch.embd);
    if (batch.pos)    free(batch.pos);
    if (batch.seq_id) free(batch.seq_id);
    if (batch.logits) free(batch.logits);
}

// Positive return values does not mean a fatal error, but rather a warning.
//   0 - success
// < 0 - error
int gpt2_decode(
        struct gpt2_model &  model,
        ggml_gallocr_t       allocr,
        struct gpt2_batch    batch,
        int                  n_threads,
        std::vector<float> & logits) {
    const int32_t n_tokens = batch.n_tokens;
    const auto &  hparams  = model.hparams;
    const int     n_vocab  = hparams.n_vocab;

    if (n_tokens == 0) {
        printf("%s: n_tokens == 0", __func__);
        return -1;
    }

    GGML_ASSERT((!batch.token && batch.embd) || (batch.token && !batch.embd));

    auto & cache = model.kv_cache;

    for (int i = 0; i < n_tokens; i++) {
        cache.cells[cache.head + i].pos = batch.pos[i];
        cache.cells[cache.head + i].seq_id.insert(batch.seq_id[i]);
    }

    cache.n = cache.head + n_tokens;

    struct ggml_cgraph * gf = gpt2_graph(model, batch, false);

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    // set the graph inputs
    if (batch.token) {
        struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
        ggml_backend_tensor_set(inp_tokens, batch.token, 0, n_tokens*ggml_element_size(inp_tokens));

        struct ggml_tensor * position = ggml_graph_get_tensor(gf, "position");
        for (int i = 0; i < n_tokens; ++i) {
            int32_t v = batch.pos[i];
            ggml_backend_tensor_set(position, &v, i*sizeof(int32_t), sizeof(v));
        }
    } else {
        struct ggml_tensor * embd = ggml_graph_get_tensor(gf, "embd");
        ggml_backend_tensor_set(embd, batch.embd, 0, n_tokens * hparams.n_embd * ggml_element_size(embd));
    }

    {
        struct ggml_tensor * KQ_mask = ggml_graph_get_tensor(gf, "KQ_mask");
        const auto & kv_cache = model.kv_cache;
        const int32_t n_tokens = batch.n_tokens;
        const int32_t n_kv     = kv_cache.n;

        std::vector<float> data_buf(n_kv*n_tokens);
        const float neg_inf_v = -INFINITY;

        for (int h = 0; h < 1; ++h) {
            int h_offset = h*(n_kv*n_tokens);
            for (int j = 0; j < n_tokens; ++j) {
                const gpt2_pos    pos    = batch.pos[j];
                const gpt2_seq_id seq_id = batch.seq_id[j];

                for (int i = 0; i < n_kv; ++i) {
                    if (!kv_cache.cells[i].has_seq_id(seq_id) || kv_cache.cells[i].pos > pos) {
                        data_buf[h_offset + j*n_kv + i] = neg_inf_v;
                    }
                }
            }
        }

        ggml_backend_tensor_set(KQ_mask, data_buf.data(), 0, data_buf.size() * sizeof(float));
    }

    // run the computation
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }
#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend)) {
        ggml_backend_metal_set_n_cb(model.backend, n_threads);
    }
#endif
    ggml_backend_graph_compute(model.backend, gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    // in this case, the output tensor is the last one in the graph
    struct ggml_tensor * inpL = gf->nodes[gf->n_nodes - 1];

    if (batch.logits) {
        // return logits for all tokens
        logits.resize(n_vocab*n_tokens);
        for (int32_t i = 0; i < n_tokens; i++) {
            if (batch.logits[i] == 0) {
                continue;
            }
            ggml_backend_tensor_get(inpL, logits.data() + n_vocab*i, n_vocab*i*sizeof(float), sizeof(float)*n_vocab);
        }
    } else {
        // return result just for the last token
        logits.resize(n_vocab);
        ggml_backend_tensor_get(inpL, logits.data(), (n_vocab*(n_tokens-1))*sizeof(float), sizeof(float)*n_vocab);
    }

    // update the kv ring buffer
    cache.head += n_tokens;

    // ensure kv cache head points to a valid index.
    if (cache.head >= cache.size) {
        printf("%s: cache.head >= cache.size\n", __func__);
        return -2;
    }

    return 0;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    const int64_t t_main_start_us = ggml_time_us();

    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    printf("%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.prompt.empty()) {
        params.prompt = gpt_random_prompt(rng);
    }

    int64_t t_load_us = 0;

    gpt_vocab vocab;
    gpt2_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!gpt2_model_load(params.model, model, vocab, params.n_ctx, params.n_gpu_layers)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;

        test_gpt_tokenizer(vocab, params.token_test);
    }

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);

    const int n_parallel = params.n_parallel;
    const int n_batch_max = std::max(embd_inp.size(), (size_t)n_parallel);

    // create a gpt2_batch
    // we use this object to submit token data for decoding
    gpt2_batch batch = gpt2_batch_init(n_batch_max, 0);

    // prepare required memory and allocate the compute buffer
    ggml_gallocr_t allocr = NULL;
    {
        // create an allocator to measure the memory usage
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        // create the worst case graph for memory usage estimation
        batch.n_tokens = n_batch_max;
        struct ggml_cgraph * gf = gpt2_graph(model, batch, true);

        // pre-allocate the compute buffer for the worst case (optional)
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);
        fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0/1024.0);
    }

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // evaluate the initial prompt
    batch.n_tokens = embd_inp.size();

    for (int32_t i = 0; i < batch.n_tokens; i++) {
        batch.token[i]  = embd_inp[i];
        batch.pos[i]    = i;
        batch.seq_id[i] = 0;
        batch.logits[i] = false;
    }

    // gpt2_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (gpt2_decode(model, allocr, batch, params.n_threads, logits) != 0) {
        printf("%s: gpt2_decode() failed\n", __func__);
        return 1;
    }

    // assign the system KV cache to all parallel sequences
    // this way, the parallel sequences will "reuse" the prompt tokens without having to copy them
    for (int32_t i = 1; i < n_parallel; ++i) {
        gpt2_kv_cache_seq_cp(model.kv_cache, 0, i, 0, batch.n_tokens);
    }

    if (n_parallel > 1) {
        printf("\n\n%s: generating %d sequences ...\n", __func__, n_parallel);
    }

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());

    printf("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    printf("%s: number of tokens in prompt = %zu, first 8 tokens: ", __func__, embd_inp.size());
    for (int i = 0; i < std::min(8, (int) embd_inp.size()); i++) {
        printf("%d ", embd_inp[i]);
    }
    printf("\n\n");

    std::vector<gpt_vocab::token> streams(n_parallel);

    // remember the batch index of the last token for each parallel sequence
    // we need this to determine which logits to sample from
    std::vector<int32_t> i_batch(n_parallel, batch.n_tokens - 1);

    int n_cur     = batch.n_tokens;
    int n_len     = batch.n_tokens + params.n_predict;
    int n_decoded = 0;

    const int   n_vocab = model.hparams.n_vocab;
    const int   top_k = params.top_k;
    const float top_p = params.top_p;
    const float temp  = params.temp;

    while (n_cur < n_len) {
        batch.n_tokens = 0;

        for (int32_t i = 0; i < n_parallel; ++i) {
            if (i_batch[i] < 0) {
                // the stream has already finished
                continue;
            }

            auto * logits_i = logits.data() + i_batch[i]*n_vocab;

            gpt_vocab::id id = 0;
            {
                const int64_t t_start_sample_us = ggml_time_us();

                id = gpt_sample_top_k_top_p(vocab, logits_i, top_k, top_p, temp, rng);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // is it an end of stream? -> mark the stream as finished
            if ((!params.ignore_eos && id == 50256) || n_cur == n_len - 1) {
                i_batch[i] = -1;
                printf("\n");
                if (n_parallel > 1) {
                    printf("%s: stream %d finished at n_cur = %d", __func__, i, n_cur);
                }

                continue;
            }

            auto& token = vocab.id_to_token[id];
            if (n_parallel == 1) {
                printf("%s", token.c_str());
                fflush(stdout);
            }

            streams[i] += token;

            // push this new token for next evaluation
            batch.token [batch.n_tokens] = id;
            batch.pos   [batch.n_tokens] = n_cur;
            batch.seq_id[batch.n_tokens] = i;
            batch.logits[batch.n_tokens] = true;

            i_batch[i] = batch.n_tokens;

            batch.n_tokens += 1;

            n_decoded += 1;
        }

        // all streams are finished
        if (batch.n_tokens == 0) {
            break;
        }

        n_cur += 1;

        {
            const int64_t t_start_us = ggml_time_us();

            // evaluate the current batch with the transformer model
            int ret_code = gpt2_decode(model, allocr, batch, params.n_threads, logits);
            if (ret_code != 0) {
                fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, ret_code);
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }
    }

    if (n_parallel > 1) {
        printf("\n");

        for (int32_t i = 0; i < n_parallel; ++i) {
            printf("sequence %d:\n\n%s%s\n\n", i, params.prompt.c_str(), streams[i].c_str());
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s:     n_decoded = %8d\n",      __func__, n_decoded);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        printf("%s:  predict time = %8.2f ms\n", __func__, t_predict_us/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    gpt2_batch_free(batch);
    ggml_free(model.ctx_w);

    ggml_gallocr_free(allocr);
    ggml_backend_buffer_free(model.buffer_w);
    ggml_backend_buffer_free(model.kv_cache.buffer);
    ggml_backend_free(model.backend);

    return 0;
}
