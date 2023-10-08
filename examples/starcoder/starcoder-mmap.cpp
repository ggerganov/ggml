#include "ggml/ggml.h"

#include "common.h"
#include "common-ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#if !defined(_WIN32)
// mmap
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#else
#define NOMINMAX
#include <Windows.h>
#endif

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_CLBLAST
#include "ggml-opencl.h"
#endif

// default hparams (GPT-2 117M)
// https://huggingface.co/bigcode/gpt_bigcode-santacoder/blob/main/config.json
struct starcoder_hparams {
    int32_t n_vocab = 49280;
    int32_t n_ctx   = 2048;
    int32_t n_embd  = 2048;
    int32_t n_head  = 16;
    int32_t n_layer = 24;
    int32_t ftype   = 1;
    float   eps     = 1e-5f;
};

struct starcoder_layer {
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

struct llama_buffer {
    uint8_t * addr = NULL;
    size_t size = 0;

    llama_buffer() = default;

    void resize(size_t len) {
#ifdef GGML_USE_METAL
        free(addr);
        int result = posix_memalign((void **) &addr, getpagesize(), len);
        if (result == 0) {
            memset(addr, 0, len);
        }
        else {
            addr = NULL;
        }
#else
        delete[] addr;
        addr = new uint8_t[len];
#endif
        size = len;
    }

    ~llama_buffer() {
#ifdef GGML_USE_METAL
        free(addr);
#else
        delete[] addr;
#endif
        addr = NULL;
    }

    // disable copy and move
    llama_buffer(const llama_buffer&) = delete;
    llama_buffer(llama_buffer&&) = delete;
    llama_buffer& operator=(const llama_buffer&) = delete;
    llama_buffer& operator=(llama_buffer&&) = delete;
};


struct kv_cache {
    struct ggml_tensor * k;
    struct ggml_tensor * v;

    struct ggml_context * ctx = NULL;

    //std::vector<uint8_t> buf;
    llama_buffer buf;

    int n;
};

struct starcoder_model {
    starcoder_hparams hparams;

    // normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * wte;     // position embedding
    struct ggml_tensor * wpe;     //    token embedding
    struct ggml_tensor * lm_head; // language model head

    std::vector<starcoder_layer> layers;

    // key + value memory
    //struct ggml_tensor * memory_k;
    //struct ggml_tensor * memory_v;
    struct kv_cache cache;

    // model memory mapped file
    void * mm_addr = NULL;
    uint64_t mm_length = 0;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// From PR #613 (https://github.com/ggerganov/llama.cpp/pull/613)
static void *mmap_file(const char *fname, uint64_t *mm_length) {
#if defined(_WIN32) && !defined(_POSIX_MAPPED_FILES)
    HANDLE hFile = CreateFileA(fname,
                               GENERIC_READ,
                               FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               NULL,
                               OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL | FILE_ATTRIBUTE_NOT_CONTENT_INDEXED,
                               NULL);
    if (hFile == INVALID_HANDLE_VALUE) return 0;
    LARGE_INTEGER fileSize;
    fileSize.QuadPart = -1;
    GetFileSizeEx(hFile, &fileSize);
    int64_t length = fileSize.QuadPart;
    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(hFile);
    if (!hMapping) return 0;
    void *addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMapping);
    if (!addr) return 0;
#else
    int fd = open(fname, O_RDONLY);
    if (fd == -1) return 0;
    int64_t length = lseek(fd, 0, SEEK_END);
    void *addr = mmap(NULL, length, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) return 0;
#endif
    *mm_length = length;
    return addr;
}

static void munmap_file(void * addr, size_t length) {
#if defined(_WIN32) && !defined(_POSIX_MAPPED_FILES)
    UnmapViewOfFile(addr);
#else
    munmap(addr, length);
#endif
}

// load the model's weights from a file
bool starcoder_model_load(const std::string & fname, starcoder_model & model, gpt_vocab & vocab, int32_t n_gpu_layers) {
    printf("%s: loading model from '%s'\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    std::vector<char> f_buf(1024*1024);
    fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());


    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        //if (magic != 0x67676a74) {
        if (magic != 0x67676d6c) {
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

            // if (i < 10) fprintf(stderr, "%.s: vocab[%d] = '%s'\n", __func__, i, word.c_str());
        }

        // Add StarChat special tokens.
        for (std::string token : {
                "<|system|>",
                "<|user|>",
                "<|assistant|>",
                "<|end|>",
                }) {
            if (vocab.token_to_id.find(token) != vocab.token_to_id.end()) {
                vocab.add_special_token(token);
            }
        }
    }

    char *mm_addr = NULL;
    model.mm_addr = mmap_file(fname.c_str(), &model.mm_length);
    if (model.mm_addr == NULL) {
        fprintf(stderr, "%s: failed to mmap '%s'\n", __func__, fname.c_str());
        return false;
    }
    mm_addr = (char *)model.mm_addr;
    fprintf(stderr, "%s: ggml map size = %6.2f MB\n", __func__, model.mm_length/(1024.0*1024.0));

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

        const int n_layer = hparams.n_layer;


        /*
        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        const int head_dim = n_embd / hparams.n_head;
        const int kv_heads = hparams.n_head; // 1 if MQA else hparams.n_head
        const int kv_dim   = kv_heads * head_dim;


        ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // ln_f_g
        ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // ln_f_b

        ctx_size += n_vocab*n_embd*ggml_type_sizef(wtype);         // wte
        ctx_size +=   n_ctx*n_embd*ggml_type_sizef(GGML_TYPE_F32); // wpe
        ctx_size += n_vocab*n_embd*ggml_type_sizef(wtype);         // lm_head

        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_1_g
        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_1_b

        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_2_g
        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ln_2_b

        ctx_size += n_layer*((n_embd + 2*kv_dim)*n_embd*ggml_type_sizef(wtype));         // c_attn_attn_w // TODO:
        ctx_size += n_layer*(       (n_embd + 2*kv_dim)*ggml_type_sizef(GGML_TYPE_F32)); // c_attn_attn_b

        ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype));           // c_attn_proj_w
        ctx_size += n_layer*(       n_embd*ggml_type_sizef(GGML_TYPE_F32));   // c_attn_proj_b

        ctx_size += n_layer*(4*n_embd*n_embd*ggml_type_sizef(wtype));         // c_mlp_fc_w
        ctx_size += n_layer*(       4*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_mlp_fc_b

        ctx_size += n_layer*(4*n_embd*n_embd*ggml_type_sizef(wtype));         // c_mlp_proj_w
        ctx_size += n_layer*(         n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_mlp_proj_b

        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_k
        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_v
        */

        ctx_size += (6 + 12*n_layer)*512; // object overhead

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
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
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        const int head_dim = n_embd / hparams.n_head;
        const int kv_heads = hparams.n_head; // 1 if MQA else hparams.n_head
        const int kv_dim   = kv_heads * head_dim;

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

            layer.c_attn_attn_w = ggml_new_tensor_2d(ctx, wtype,           n_embd, n_embd + 2*kv_dim);
            layer.c_attn_attn_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd + 2*kv_dim);

            layer.c_attn_proj_w = ggml_new_tensor_2d(ctx, wtype,           n_embd, n_embd);
            layer.c_attn_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.c_mlp_fc_w    = ggml_new_tensor_2d(ctx, wtype,           n_embd, 4*n_embd); //TODO: 4*n_embd = config.n_inner
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

    // key + value memory
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;

        const int n_mem      = n_layer*n_ctx;
        const int n_elements = n_embd*n_mem;

        model.cache.buf.resize(2u*n_elements*ggml_type_size(GGML_TYPE_F16) + 2u*1024*1024);

        struct ggml_init_params c_params;
        c_params.mem_size   = model.cache.buf.size;
        c_params.mem_buffer = model.cache.buf.addr;
        c_params.no_alloc   = false;

        model.cache.ctx = ggml_init(c_params);

        if (!model.cache.ctx) {
            fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
            return false;
        }

        model.cache.k = ggml_new_tensor_1d(model.cache.ctx, GGML_TYPE_F16, n_elements);
        model.cache.v = ggml_new_tensor_1d(model.cache.ctx, GGML_TYPE_F16, n_elements);

        const size_t memory_size = ggml_nbytes(model.cache.k) + ggml_nbytes(model.cache.v);

        printf("%s: kv_cache memory size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }

    // load weights
    {
        size_t total_size = 0;

        bool has_lm_head = false;

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

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.data(), (int) tensor->ne[0], (int) tensor->ne[1], ne[0], ne[1]);
                return false;
            }
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file. got %d, expected %d\n",
                        __func__, name.data(), (int) ggml_nelements(tensor), nelements);
                return false;
            }

            // for debugging
            if (0) {
                printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", name.data(), ne[0], ne[1], ggml_type_name(ggml_type(ttype)), ggml_nbytes(tensor)/1024.0/1024.0, ggml_nbytes(tensor));
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            // mmap
            size_t offset = fin.tellg();
            size_t tensor_data_size = ggml_nbytes(tensor);
            //offset = (offset + 31) & -32;
            tensor->data = mm_addr + offset;
            fin.seekg(offset + tensor_data_size);
            total_size += tensor_data_size;

            // GPT-2 models share the WTE tensor as the LM head
            if (name == "model/wte" && has_lm_head == false) {
                // Dont know if this is required, test models have an lm_head
                model.lm_head->data = tensor->data;
            }

            if (name == "model/lm_head") {
                has_lm_head = true;
            }
        }

        printf("%s: model size  = %8.2f MB\n", __func__, total_size/1024.0/1024.0);
    }

    fin.close();

#ifdef GGML_USE_CUBLAS
    {
        const auto & hparams = model.hparams;
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));

        fprintf(stderr, "%s: [cublas] offloading %d layers to GPU\n", __func__, n_gpu);

        size_t vram_total = 0;

        for (int i = 0; i < n_gpu; ++i) {
            const auto & layer = model.layers[i];

            layer.c_attn_attn_w->backend = GGML_BACKEND_GPU;
            ggml_cuda_transform_tensor((uint8_t *)layer.c_attn_attn_w->data, layer.c_attn_attn_w); vram_total += ggml_nbytes(layer.c_attn_attn_w);

            layer.c_attn_proj_w->backend = GGML_BACKEND_GPU;
            ggml_cuda_transform_tensor((uint8_t *)layer.c_attn_proj_w->data, layer.c_attn_proj_w); vram_total += ggml_nbytes(layer.c_attn_proj_w);

            layer.c_mlp_fc_w->backend = GGML_BACKEND_GPU;
            ggml_cuda_transform_tensor((uint8_t *)layer.c_mlp_fc_w->data, layer.c_mlp_fc_w); vram_total += ggml_nbytes(layer.c_mlp_fc_w);

            layer.c_mlp_proj_w->backend = GGML_BACKEND_GPU;
            ggml_cuda_transform_tensor((uint8_t *)layer.c_mlp_proj_w->data, layer.c_mlp_proj_w); vram_total += ggml_nbytes(layer.c_mlp_proj_w);
        }

        ggml_cuda_set_scratch_size(0); // disable scratch

        //if (n_gpu_layers > (int) hparams.n_layer) {
        //    fprintf(stderr, "%s: [cublas] offloading output layer to GPU\n", __func__);
        //    ggml_cuda_transform_tensor(model.output); vram_total += ggml_nbytes(model.output);
        //}

        fprintf(stderr, "%s: [cublas] total VRAM used: %zu MB\n", __func__, vram_total / 1024 / 1024);
    }
#elif defined(GGML_USE_CLBLAST)
    //From koboldcpp
    {
        const auto & hparams = model.hparams;
        size_t vram_total = 0;
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));
        fprintf(stderr, "%s: [opencl] offloading %d layers to GPU\n", __func__, n_gpu);
        for (int i = 0; i < n_gpu; ++i) {
            const auto & layer = model.layers[i];
            layer.c_attn_attn_w->backend = GGML_BACKEND_GPU;
            layer.c_attn_proj_w->backend = GGML_BACKEND_GPU;
            layer.c_mlp_fc_w->backend = GGML_BACKEND_GPU;
            layer.c_mlp_proj_w->backend = GGML_BACKEND_GPU;
            ggml_cl_transform_tensor(layer.c_attn_attn_w->data,layer.c_attn_attn_w); vram_total += ggml_nbytes(layer.c_attn_attn_w);
            ggml_cl_transform_tensor(layer.c_attn_proj_w->data,layer.c_attn_proj_w); vram_total += ggml_nbytes(layer.c_attn_proj_w);
            ggml_cl_transform_tensor(layer.c_mlp_fc_w->data,layer.c_mlp_fc_w); vram_total += ggml_nbytes(layer.c_mlp_fc_w);
            ggml_cl_transform_tensor(layer.c_mlp_proj_w->data,layer.c_mlp_proj_w); vram_total += ggml_nbytes(layer.c_mlp_proj_w);
        }
        fprintf(stderr, "%s: [opencl] total VRAM used: %zu MB\n", __func__, vram_total / 1024 / 1024);
    }
    #endif

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
bool starcoder_eval(
        const starcoder_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w,
              size_t                     & mem_per_token) {

    const int N = int(embd_inp.size());

    const auto & hparams = model.hparams;

    auto & cache = model.cache;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_vocab;

    // Scratch is too small for large n_batch (256)
    //static size_t buf_size = 256u*1024*1024;
    static size_t buf_size = 256u*1024*1024*2;
    static void * buf = malloc(buf_size);

    // use 2 scratch buffers
    // TODO: very hacky solution - reimplement in a more elegant way
    static size_t scratch0_size = 256u*1024*1024*2;
    static void * scratch0 = malloc(scratch0_size);

    static size_t scratch1_size = 256u*1024*1024*2;
    static void * scratch1 = malloc(scratch1_size);

    if (mem_per_token > 0 && mem_per_token*N > buf_size) {
        const size_t buf_size_new = size_t(1.1*(mem_per_token*N)); // add 10% to account for ggml object overhead
        printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

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


    memcpy(embd->data, embd_inp.data(), N*ggml_element_size(embd));

    struct ggml_tensor * position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    for (int i = 0; i < N; ++i) {
        ((int32_t *) position->data)[i] = n_past + i;
    }

    // wte + wpe
    struct ggml_tensor * inpL =
        ggml_add(ctx0,
                ggml_get_rows(ctx0, model.wte, embd),
                ggml_get_rows(ctx0, model.wpe, position));

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur;

        ggml_set_scratch(ctx0, { 0, scratch0_size, scratch0, });

        // norm
        {
            // [ 768, N]
            cur = ggml_norm(ctx0, inpL, hparams.eps);

            // cur = ln_1_g*cur + ln_1_b
            // [ 768, N]
            cur = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].ln_1_g, cur),
                        cur),
                    ggml_repeat(ctx0, model.layers[il].ln_1_b, cur));
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
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].c_attn_attn_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[il].c_attn_attn_b, cur),
                    cur);
        }

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 0*sizeof(float)*n_embd);
            struct ggml_tensor * Kcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 1*sizeof(float)*n_embd);
            struct ggml_tensor * Vcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 2*sizeof(float)*n_embd);

            // store key and value to memory
            if (N >= 1) {
                struct ggml_tensor * k = ggml_view_1d(ctx0, cache.k, N*n_embd, (ggml_element_size(cache.k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_1d(ctx0, cache.v, N*n_embd, (ggml_element_size(cache.v)*n_embd)*(il*n_ctx + n_past));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            // [64, N, 12]
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        ggml_cpy(ctx0,
                            Qcur,
                            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N)),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            // [64, n_past + N, 12]
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, cache.k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(cache.k)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3); //TODO: need to be tiled

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
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q); //TODO: check if it broadcasts

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // [n_past + N, N, 12]
            struct ggml_tensor * KQ_scaled =
                ggml_scale_inplace(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
                        );

            // KQ_masked = mask_past(KQ_scaled)
            // [n_past + N, N, 12]
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            // [n_past + N, N, 12]
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            // [n_past + N, 64, 12]
            struct ggml_tensor * V_trans =
                ggml_cpy(ctx0,
                        ggml_permute(ctx0,
                            ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, cache.v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(cache.v)*n_embd),
                                n_embd/n_head, n_head, n_past + N),
                            1, 2, 0, 3),
                        ggml_new_tensor_3d(ctx0, cache.v->type, n_past + N, n_embd/n_head, n_head));

            // KQV = transpose(V) * KQ_soft_max
            // [64, N, 12]
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // [64, 12, N]
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // [768, N]
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
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
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].c_attn_proj_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[il].c_attn_proj_b, cur),
                    cur);
        }

        // add the input
        cur = ggml_add(ctx0, cur, inpL);

        struct ggml_tensor * inpFF = cur;

        ggml_set_scratch(ctx0, { 0, scratch1_size, scratch1, });

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF, hparams.eps);

                // cur = ln_2_g*cur + ln_2_b
                // [ 768, N]
                cur = ggml_add(ctx0,
                        ggml_mul(ctx0,
                            ggml_repeat(ctx0, model.layers[il].ln_2_g, cur),
                            cur),
                        ggml_repeat(ctx0, model.layers[il].ln_2_b, cur));
            }

            // fully connected
            // [3072, 768] - model.layers[il].c_mlp_fc_w
            // [3072,   1] - model.layers[il].c_mlp_fc_b
            // [ 768,   N] - cur (in)
            // [3072,   N] - cur (out)
            //
            // cur = fc_w*cur + fc_b
            // [3072, N]
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].c_mlp_fc_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[il].c_mlp_fc_b, cur),
                    cur);

            // GELU activation
            // [3072, N]
            cur = ggml_gelu(ctx0, cur);

            // projection
            // [ 768, 3072] - model.layers[il].c_mlp_proj_w
            // [ 768,    1] - model.layers[il].c_mlp_proj_b
            // [3072,    N] - cur (in)
            // [ 768,    N] - cur (out)
            //
            // cur = proj_w*cur + proj_b
            // [768, N]
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].c_mlp_proj_w,
                    cur);

            cur = ggml_add(ctx0,
                    ggml_repeat(ctx0, model.layers[il].c_mlp_proj_b, cur),
                    cur);
        }

        // input for next layer
        inpL = ggml_add(ctx0, cur, inpFF);
    }

    ggml_set_scratch(ctx0, { 0, scratch0_size, scratch0, });

    // norm
    {
        // [ 768, N]
        inpL = ggml_norm(ctx0, inpL, hparams.eps);

        // inpL = ln_f_g*inpL + ln_f_b
        // [ 768, N]
        inpL = ggml_add(ctx0,
                ggml_mul(ctx0,
                    ggml_repeat(ctx0, model.ln_f_g, inpL),
                    inpL),
                ggml_repeat(ctx0, model.ln_f_b, inpL));
    }

    ggml_set_scratch(ctx0, { 0, 0, nullptr, });

    // inpL = WTE * inpL
    // [ 768, 50257] - model.lm_head
    // [ 768, N]     - inpL
    inpL = ggml_mul_mat(ctx0, model.lm_head, inpL);

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

    // return result just for the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }
    //printf("used_mem = %zu MB\n", ggml_used_mem(ctx0)/(1024*1024));

    ggml_free(ctx0);

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
        params.seed = int(time(NULL));
    }

    printf("%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.prompt.empty()) {
        params.prompt = gpt_random_prompt(rng);
    }

    int64_t t_load_us = 0;

    gpt_vocab vocab;
    starcoder_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!starcoder_model_load(params.model, model, vocab, params.n_gpu_layers)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;

        test_gpt_tokenizer(vocab, params.token_test);
    }

    int n_past = 0;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());

    printf("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
    for (size_t i = 0; i < embd_inp.size(); i++) {
        printf("%s: token[%zu] = %6d, %s\n", __func__, i, embd_inp[i], vocab.id_to_token.at(embd_inp[i]).c_str());
    }
    printf("\n\n");

    // Handle StarChat "<|end|>" token.
    gpt_vocab::id starchat_end_token = -1;
    {
        const auto it = vocab.token_to_id.find("<|end|>");
        if (it != vocab.token_to_id.end()) {
            starchat_end_token = it->second;
        }
    }

    // submit the input prompt token-by-token
    // this reduces the memory usage during inference, at the cost of a bit of speed at the beginning
    std::vector<gpt_vocab::id> embd;

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    printf("Calling starcoder_eval\n");
    starcoder_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);

    for (size_t i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!starcoder_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) {
                printf("Failed to predict\n");
                return 1;
            }

            // Should input processing count towards t_predict?
            if (i > embd_inp.size()) {
                t_predict_us += ggml_time_us() - t_start_us;
            }
        }

        n_past += int(embd.size());
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
            i += int(embd.size()) - 1;
        }

        // display text
        for (auto id : embd) {
            printf("%s", vocab.id_to_token[id].c_str());
        }
        fflush(stdout);

        // check if model is santacoder
        if (model.hparams.n_layer <= 30 && embd.back() == 49152) {
            break;
        }
        // check if model is starcoder
        else if (embd.back() == 0) { //TODO: this is only for starcoder
            break;
        }
        // Handle StarChat "<|end|>" token.
        else if (embd.back() == starchat_end_token) {
            //break;
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        //Shouldnt the input prompt be subracted?
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/(n_past - embd_inp.size()));
        //printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);

        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    ggml_free(model.ctx);

    if (model.mm_addr) {
	    munmap_file(model.mm_addr, model.mm_length);
    }

    return 0;
}
