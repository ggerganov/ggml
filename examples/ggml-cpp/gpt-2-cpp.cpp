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
#include "ggml-cpp.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <cstdarg>


std::string format(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);

    // Get the required size
    int size = std::vsnprintf(nullptr, 0, fmt, args);
    va_end(args);

    if(size <= 0) {
        return "";
    }

    std::string result(size, '\0');

    va_start(args, fmt);
    std::vsnprintf(&result[0], size + 1, fmt, args);
    va_end(args);

    return result;
}

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

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
    gpt2_layer(ggml_type wtype, int n_embd) {
        ln_1_g        = ggml::tensor(GGML_TYPE_F32,   n_embd);
        ln_1_b        = ggml::tensor(GGML_TYPE_F32,   n_embd);
        ln_2_g        = ggml::tensor(GGML_TYPE_F32,   n_embd);
        ln_2_b        = ggml::tensor(GGML_TYPE_F32,   n_embd);
        c_attn_attn_w = ggml::tensor(wtype,           n_embd, 3*n_embd);
        c_attn_attn_b = ggml::tensor(GGML_TYPE_F32, 3*n_embd);
        c_attn_proj_w = ggml::tensor(wtype,           n_embd, n_embd);
        c_attn_proj_b = ggml::tensor(GGML_TYPE_F32,   n_embd);
        c_mlp_fc_w    = ggml::tensor(wtype,           n_embd, 4*n_embd);
        c_mlp_fc_b    = ggml::tensor(GGML_TYPE_F32, 4*n_embd);
        c_mlp_proj_w  = ggml::tensor(wtype,         4*n_embd, n_embd);
        c_mlp_proj_b  = ggml::tensor(GGML_TYPE_F32,   n_embd);
    }

    // normalization
    ggml::tensor ln_1_g;
    ggml::tensor ln_1_b;

    ggml::tensor ln_2_g;
    ggml::tensor ln_2_b;

    // attention
    ggml::tensor c_attn_attn_w;
    ggml::tensor c_attn_attn_b;

    ggml::tensor c_attn_proj_w;
    ggml::tensor c_attn_proj_b;

    // mlp
    ggml::tensor c_mlp_fc_w;
    ggml::tensor c_mlp_fc_b;

    ggml::tensor c_mlp_proj_w;
    ggml::tensor c_mlp_proj_b;
};

struct gpt2_model {
    gpt2_model(ggml_type wtype, gpt2_hparams hparams) : hparams(hparams) {
        ctx = ggml::context(ggml_tensor_overhead()*(2 + 6 + 12*hparams.n_layer), NULL, true);
        ggml::context_guard ctx_guard(ctx);

        ln_f_g   = ggml::tensor(GGML_TYPE_F32, hparams.n_embd);
        ln_f_b   = ggml::tensor(GGML_TYPE_F32, hparams.n_embd);

        wte      = ggml::tensor(wtype,         hparams.n_embd, hparams.n_vocab);
        wpe      = ggml::tensor(GGML_TYPE_F32, hparams.n_embd, hparams.n_ctx);
        lm_head  = ggml::tensor(wtype,         hparams.n_embd, hparams.n_vocab);

        layers.reserve(hparams.n_layer);
        for (int i = 0; i < hparams.n_layer; ++i) {
            layers.emplace_back(wtype, hparams.n_embd);
        }

        memory_k = ggml::tensor(GGML_TYPE_F32, hparams.n_embd*hparams.n_layer*hparams.n_ctx);
        memory_v = ggml::tensor(GGML_TYPE_F32, hparams.n_embd*hparams.n_layer*hparams.n_ctx);

        // map by name
        tensors["model/ln_f/g"] = &ln_f_g;
        tensors["model/ln_f/b"] = &ln_f_b;

        tensors["model/wte"]     = &wte;
        tensors["model/wpe"]     = &wpe;
        tensors["model/lm_head"] = &lm_head;

        for (int i = 0; i < hparams.n_layer; ++i) {
            gpt2_layer & layer = layers[i];

            // map by name
            tensors["model/h" + std::to_string(i) + "/ln_1/g"]        = &layer.ln_1_g;
            tensors["model/h" + std::to_string(i) + "/ln_1/b"]        = &layer.ln_1_b;

            tensors["model/h" + std::to_string(i) + "/ln_2/g"]        = &layer.ln_2_g;
            tensors["model/h" + std::to_string(i) + "/ln_2/b"]        = &layer.ln_2_b;

            tensors["model/h" + std::to_string(i) + "/attn/c_attn/w"] = &layer.c_attn_attn_w;
            tensors["model/h" + std::to_string(i) + "/attn/c_attn/b"] = &layer.c_attn_attn_b;

            tensors["model/h" + std::to_string(i) + "/attn/c_proj/w"] = &layer.c_attn_proj_w;
            tensors["model/h" + std::to_string(i) + "/attn/c_proj/b"] = &layer.c_attn_proj_b;

            tensors["model/h" + std::to_string(i) + "/mlp/c_fc/w"]    = &layer.c_mlp_fc_w;
            tensors["model/h" + std::to_string(i) + "/mlp/c_fc/b"]    = &layer.c_mlp_fc_b;

            tensors["model/h" + std::to_string(i) + "/mlp/c_proj/w"]  = &layer.c_mlp_proj_w;
            tensors["model/h" + std::to_string(i) + "/mlp/c_proj/b"]  = &layer.c_mlp_proj_b;
        }
    }

    gpt2_hparams hparams;
    ggml::context ctx;

    // normalization
    ggml::tensor ln_f_g;
    ggml::tensor ln_f_b;

    ggml::tensor wte;     // position embedding
    ggml::tensor wpe;     //    token embedding
    ggml::tensor lm_head; // language model head

    std::vector<gpt2_layer> layers;

    // key + value memory
    ggml::tensor memory_k;
    ggml::tensor memory_v;

    ggml::backend backend;
    ggml::backend_buffer buffer_w;
    ggml::backend_buffer buffer_kv;

    std::map<std::string, ggml::tensor*> tensors;
};

// load the model's weights from a file
gpt2_model gpt2_model_load(const std::string & fname, gpt_vocab & vocab, int n_gpu_layers) {
    printf("%s: loading model from '%s'\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        throw std::runtime_error(format("failed to open '%s'", fname.c_str()));
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            throw std::runtime_error(format("invalid model file '%s' (bad magic)", fname.c_str()));
        }
    }

    // load hparams
    gpt2_hparams hparams;
    {
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

        if (n_vocab != hparams.n_vocab) {
            throw std::runtime_error(format("invalid model file '%s' (bad vocab size %d != %d)", fname.c_str(), n_vocab, hparams.n_vocab));
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
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype) (hparams.ftype));
    if (wtype == GGML_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
                __func__, fname.c_str(), hparams.ftype);
    }

    // initialize the model object
    gpt2_model model(wtype, hparams);

    // initialize the backend
#ifdef GGML_USE_CUBLAS
    if (n_gpu_layers > 0) {
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (n_gpu_layers > 0) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        ggml_metal_log_set_callback(ggml_log_callback_default, nullptr);
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
        throw std::runtime_error("ggml_backend_cpu_init() failed");
    }

    // calculate the size of the backend buffer
    size_t buffer_size = 0;
    {
        for (auto it : model.tensors) {
            buffer_size += it.second->nbytes() + 128;
        }
        printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
        printf("%s: backend buffer size = %6.2f MB\n", __func__, buffer_size/(1024.0*1024.0));
    }

    // allocate weights buffer
    model.buffer_w = model.backend.alloc_buffer(buffer_size);

    // key + value memory
    {
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_mem   = n_layer*n_ctx;

        const size_t memory_size = model.memory_k.nbytes() + model.memory_v.nbytes();

        printf("%s: memory size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);

        // create a backend buffer (can be in host or device memory)
        model.buffer_kv =  model.backend.alloc_buffer(memory_size + 256);

        // allocate the tensors into the backend buffer
        {
            ggml::allocr alloc(model.buffer_kv);

            // this updates the pointers in the tensors to point to the correct location in the buffer
            // this is necessary since the ggml_context is .no_alloc == true
            // note that the buffer can actually be a device buffer, depending on the backend
            alloc.alloc(model.memory_k);
            alloc.alloc(model.memory_v);
        }
    }

    // load weights
    {
        ggml::allocr alloc(model.buffer_w);

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
                throw std::runtime_error(format("unknown tensor '%s' in model file", name.c_str()));
            }

            auto & tensor = *model.tensors[name];
            tensor.set_name(name);
            if (tensor.nelements() != nelements) {
                throw std::runtime_error(format("tensor '%s' has wrong size in model file", name.c_str()));
            }

            if (tensor.ne(0) != ne[0] || tensor.ne(1) != ne[1]) {
                throw std::runtime_error(format("tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]",
                    name.c_str(), (int) tensor.ne(0), (int) tensor.ne(1), ne[0], ne[1]));
            }

            // for debugging
            if (0) {
                printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", name.c_str(), ne[0], ne[1], ggml_type_name(ggml_type(ttype)), tensor.nbytes()/1024.0/1024.0, tensor.nbytes());
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements*bpe)/ggml_blck_size(tensor.type()) != tensor.nbytes()) {
                throw std::runtime_error(format("tensor '%s' has wrong size in model file: got %zu, expected %zu",
                    name.c_str(), tensor.nbytes(), nelements*bpe));
            }

            alloc.alloc(tensor);

            if (ggml_backend_is_cpu  (model.backend.get())
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend.get())
#endif
                ) {
                // for the CPU and Metal backend, we can read directly into the tensor
                fin.read(reinterpret_cast<char *>(tensor.data()), tensor.nbytes());
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(tensor.nbytes());
                fin.read(read_buf.data(), tensor.nbytes());
                tensor.backend_set(read_buf.data(), 0, tensor.nbytes());
            }

            // GPT-2 models share the WTE tensor as the LM head
            if (name == "model/wte" && has_lm_head == false) {
                alloc.alloc(model.lm_head);
                tensor.backend_copy(model.lm_head);
                //model.lm_head = tensor;
            }

            if (name == "model/lm_head") {
                has_lm_head = true;
            }

            total_size += tensor.nbytes();
        }

        printf("%s: model size  = %8.2f MB\n", __func__, total_size/1024.0/1024.0);
    }

    fin.close();

    return model;
}

// build the computation graph
struct ggml::graph gpt2_graph(
        gpt2_model & model,
        ggml::allocr & allocr,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp) {
    const int N = embd_inp.size();

    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;

    // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
    static size_t buf_size = ggml_tensor_overhead()*GGML_MAX_NODES + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    ggml::context ctx0(buf_size, buf.data(), true);

    ggml::context_guard ctx_guard(ctx0);

    ggml::graph gf;

    ggml::tensor embd(GGML_TYPE_I32, N);
    allocr.alloc(embd);

    // avoid writing to tensors if we are only measuring the memory usage
    if (!allocr.is_measure()) {
        embd.backend_set(embd_inp.data(), 0, N*embd.element_size());
    }

    ggml::tensor position(GGML_TYPE_I32, N);
    allocr.alloc(position);
    if (!allocr.is_measure()) {
        for (int i = 0; i < N; ++i) {
            int32_t v = n_past + i;
            position.backend_set(&v, i*sizeof(int32_t), sizeof(v));
        }
    }

    ggml::tensor KQ_scale(GGML_TYPE_F32);
    allocr.alloc(KQ_scale);
    if (!allocr.is_measure()) {
        float s = 1.0f/sqrtf(float(n_embd)/n_head);
        KQ_scale.backend_set(&s, 0, sizeof(s));
    }

    // wte + wpe
    ggml::tensor inpL = get_rows(model.wte, embd) + get_rows(model.wpe, position);

    for (int il = 0; il < n_layer; ++il) {
        ggml::tensor cur;

        // norm
        {
            // [ 768, N]
            cur = norm(inpL, hparams.eps);

            // [ 768, N]
            cur = cur*model.layers[il].ln_1_g + model.layers[il].ln_1_b;
        }

        // attn
        // [2304, 768] - model.layers[il].c_attn_attn_w
        // [2304,   1] - model.layers[il].c_attn_attn_b
        // [ 768,   N] - cur (in)
        // [2304,   N] - cur (out)
        //
        // cur = attn_w*cur + attn_b
        // [2304, N]
        {
            cur = mul_mat(model.layers[il].c_attn_attn_w, cur) + model.layers[il].c_attn_attn_b;
        }

        // self-attention
        {
            ggml::tensor Qcur = cur.view(n_embd, N, cur.nb(1), 0*sizeof(float)*n_embd);
            ggml::tensor Kcur = cur.view(n_embd, N, cur.nb(1), 1*sizeof(float)*n_embd);
            ggml::tensor Vcur = cur.view(n_embd, N, cur.nb(1), 2*sizeof(float)*n_embd);

            // store key and value to memory
            if (N >= 1) {
                ggml::tensor k = model.memory_k.view(N*n_embd, model.memory_k.element_size()*n_embd*(il*n_ctx + n_past));
                ggml::tensor v = model.memory_v.view(N*n_embd, model.memory_v.element_size()*n_embd*(il*n_ctx + n_past));

                // alternative? may be questionable use of operator overloading
                // k = Kcur;
                // v = Vcur;
                // gf.expand(k);
                // gf.expand(v);
                gf.expand(k.cpy(Kcur));
                gf.expand(v.cpy(Vcur));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            // [64, N, 12]
            ggml::tensor Q = Qcur.cont(n_embd/n_head, n_head, N).permute(0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            // [64, n_past + N, 12]
            ggml::tensor K = model.memory_k.view((n_past + N)*n_embd, il*n_ctx*model.memory_k.element_size()*n_embd)
                                .reshape(n_embd/n_head, n_head, n_past + N)
                                .permute(0, 2, 1, 3);

            // GG: flash attention
            //struct ggml_tensor * V =
            //    ggml_cpy(ctx0,
            //            ggml_permute(ctx0,
            //                ggml_reshape_3d(ctx0,
            //                    ggml_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
            //                    n_embd/n_head, n_head, n_past + N),
            //                1, 2, 0, 3),
            //            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_past + N, n_embd/n_head, n_head));

            //struct ggml_tensor * KQV = ggml_flash_attn(ctx0, Q, K, V, true);

            // K * Q
            // [n_past + N, N, 12]
            ggml::tensor KQ = mul_mat(K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // [n_past + N, N, 12]
            ggml::tensor KQ_scaled = KQ * KQ_scale;

            // KQ_masked = mask_past(KQ_scaled)
            // [n_past + N, N, 12]
            ggml::tensor KQ_masked = diag_mask_inf(KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            // [n_past + N, N, 12]
            ggml::tensor KQ_soft_max = soft_max(KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            // [n_past + N, 64, 12]
            ggml::tensor V_trans = model.memory_v.view((n_past + N)*n_embd, il*n_ctx*model.memory_v.element_size()*n_embd)
                                    .reshape(n_embd/n_head, n_head, n_past + N)
                                    .permute(1, 2, 0, 3)
                                    .cont(n_past + N, n_embd/n_head, n_head);

            // KQV = transpose(V) * KQ_soft_max
            // [64, N, 12]
            ggml::tensor KQV = mul_mat(V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // [64, 12, N]
            ggml::tensor KQV_merged = KQV.permute(0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // [768, N]
            cur = KQV_merged.cont(n_embd, N);
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
            cur = mul_mat(model.layers[il].c_attn_proj_w, cur) + model.layers[il].c_attn_proj_b;
        }

        // add the input
        cur = cur + inpL;

        ggml::tensor inpFF = cur.get();

        // feed-forward network
        {
            // norm
            {
                cur = norm(inpFF, hparams.eps);

                // cur = ln_2_g*cur + ln_2_b
                // [ 768, N]
                cur = cur*model.layers[il].ln_2_g + model.layers[il].ln_2_b;
            }

            // fully connected
            // [3072, 768] - model.layers[il].c_mlp_fc_w
            // [3072,   1] - model.layers[il].c_mlp_fc_b
            // [ 768,   N] - cur (in)
            // [3072,   N] - cur (out)
            //
            // cur = fc_w*cur + fc_b
            // [3072, N]
            cur = mul_mat(model.layers[il].c_mlp_fc_w, cur) + model.layers[il].c_mlp_fc_b;

            // GELU activation
            // [3072, N]
            cur = gelu(cur);

            // projection
            // [ 768, 3072] - model.layers[il].c_mlp_proj_w
            // [ 768,    1] - model.layers[il].c_mlp_proj_b
            // [3072,    N] - cur (in)
            // [ 768,    N] - cur (out)
            //
            // cur = proj_w*cur + proj_b
            // [768, N]
            cur = mul_mat(model.layers[il].c_mlp_proj_w, cur) + model.layers[il].c_mlp_proj_b;
        }

        // input for next layer
        inpL = cur + inpFF;
    }

    // norm
    {
        // [ 768, N]
        inpL = norm(inpL, hparams.eps);

        // inpL = ln_f_g*inpL + ln_f_b
        // [ 768, N]
        inpL = inpL*model.ln_f_g + model.ln_f_b;
    }

    // inpL = WTE * inpL
    // [ 768, 50257] - model.lm_head
    // [ 768, N]     - inpL
    inpL = mul_mat(model.lm_head, inpL);

    // logits -> probs
    //inpL = soft_max(inpL);

    gf.expand(inpL);

    return gf;
}

// evaluate the transformer
//
//   - model:     the model
//   - allocr:    ggml_allocr to use to allocate the compute buffer
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
bool gpt2_eval(
        gpt2_model & model,
        ggml::allocr & allocr,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w) {
    const int N = embd_inp.size();

    const auto & hparams = model.hparams;

    const int n_vocab = hparams.n_vocab;

    // reset the allocator to free all the memory allocated during the previous inference
    allocr.reset();

    ggml::graph gf = gpt2_graph(model, allocr, n_past, embd_inp);

    // allocate tensors
    allocr.alloc_graph(gf);

    // run the computation
    if (ggml_backend_is_cpu(model.backend.get())) {
        ggml_backend_cpu_set_n_threads(model.backend.get(), n_threads);
    }
#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend.get())) {
        ggml_backend_metal_set_n_cb(model.backend.get(), n_threads);
    }
#endif
    model.backend.graph_compute(gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    // in this case, the output tensor is the last one in the graph
    ggml::tensor inpL = gf.get_node(gf.n_nodes() - 1);

    //embd_w.resize(n_vocab*N);
    //inpL.backend_get(embd_w.data(), 0, sizeof(float)*n_vocab*N);

    // return result just for the last token
    embd_w.resize(n_vocab);
    inpL.backend_get(embd_w.data(), (n_vocab*(N-1))*sizeof(float), sizeof(float)*n_vocab);

    return true;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    const int64_t t_main_start_us = ggml_time_us();

    gpt_params params;
    params.model = "models/gpt-2-117M/ggml-model.bin";

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

    // load the model
    const int64_t t_start_us = ggml_time_us();

    // ...
    auto load_model = [](const std::string & model_file, gpt_vocab & vocab, const int n_gpu_layers) {
        try {
            return gpt2_model_load(model_file, vocab, n_gpu_layers);
        }
        catch (const std::exception & e) {
            fprintf(stderr, "%s: failed to load model: %s\n", __func__, e.what());
            exit(1);
        }
    };

    gpt2_model model = load_model(params.model, vocab, params.n_gpu_layers);

    t_load_us = ggml_time_us() - t_start_us;

    test_gpt_tokenizer(vocab, params.token_test);

    // keep this buffer alive while evaluating the model
    ggml::backend_buffer buf_compute;

    ggml::allocr allocr;
    // allocate the compute buffer
    {
         // alignment required by the backend
        size_t align = model.backend.get_alignment();
        allocr = ggml::allocr::new_measure(align);

        // create the worst case graph for memory usage estimation
        int n_tokens = std::min(model.hparams.n_ctx, params.n_batch);
        int n_past = model.hparams.n_ctx - n_tokens;
        ggml::graph gf = gpt2_graph(model, allocr, n_past, std::vector<gpt_vocab::id>(n_tokens, 0));

        // compute the required memory
        size_t mem_size = allocr.alloc_graph(gf);

        // recreate the allocator with the required memory
        buf_compute = model.backend.alloc_buffer(mem_size);
        allocr = ggml::allocr(buf_compute);

        fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0/1024.0);
    }

    int n_past = 0;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());

    printf("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    printf("%s: number of tokens in prompt = %zu, first 8 tokens: ", __func__, embd_inp.size());
    for (int i = 0; i < std::min(8, (int) embd_inp.size()); i++) {
        printf("%d ", embd_inp[i]);
    }
    printf("\n\n");

    // submit the input prompt token-by-token
    // this reduces the memory usage during inference, at the cost of a bit of speed at the beginning
    std::vector<gpt_vocab::id> embd;

    for (size_t i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!gpt2_eval(model, allocr, params.n_threads, n_past, embd, logits)) {
                printf("Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if (i >= embd_inp.size()) {
            // sample next token
            const int   top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;

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
                if (int32_t(embd.size()) >= params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        // display text
        for (auto id : embd) {
            printf("%s", vocab.id_to_token[id].c_str());
        }
        fflush(stdout);

        // end of text token
        if (embd.back() == 50256) {
            break;
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    return 0;
}
